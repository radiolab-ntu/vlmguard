"""Extract last-token block-output features from arbitrary image-text manifests."""

import argparse
from pathlib import Path

import numpy as np

from vlmguard.data import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model", required=True, help="local model directory or Hugging Face model ID")
    parser.add_argument("--output", required=True, help="output .npy file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=-1)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    stop = len(rows) if args.stop < 0 else min(args.stop, len(rows))
    if not 0 <= args.start < stop:
        parser.error("require 0 <= start < stop <= manifest length")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    device = next(model.parameters()).device
    layer_count = model.config.text_config.num_hidden_layers
    hidden = model.config.text_config.hidden_size
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(stop - args.start, layer_count, hidden))
    for position, row in enumerate(rows[args.start:stop]):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = Path(args.image_root) / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": row["question"]},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[prompt], images=images, videos=videos, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        states = outputs.hidden_states
        if len(states) != layer_count + 1:
            raise ValueError(f"expected {layer_count + 1} hidden-state tensors, got {len(states)}")
        features[position] = torch.stack([state[0, -1].float() for state in states[1:]]).cpu().numpy()
        if (position + 1) % 100 == 0:
            features.flush()
            print(f"extracted {position + 1}/{stop - args.start}", flush=True)
    features.flush()
    print(f"saved {features.shape} to {output}")


if __name__ == "__main__":
    main()
