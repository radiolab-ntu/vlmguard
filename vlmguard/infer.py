"""Score external last-token features with a trained VLMGuard classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vlmguard.train import make_classifier, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True, help="output .npy scores in input row order")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run = Path(args.run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run / "subspace.npz") as subspace:
        layer = int(subspace["layer"])
        dimension = len(subspace["mean"])
    features = np.load(args.features, mmap_mode="r")
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or layer >= features.shape[1] or features.shape[2] != dimension:
        raise ValueError("features do not match the trained layer or hidden dimension")
    model = make_classifier(dimension, int(metrics["config"]["hidden_dim"]), float(metrics["config"].get("dropout", 0.0)))
    model.load_state_dict(torch.load(run / "classifier.pt", map_location="cpu", weights_only=True))
    device = torch.device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = predict(model.to(device), features[:, layer], device, args.batch_size)
    np.save(output, scores)
    print(f"saved {len(scores)} maliciousness scores to {output}")


if __name__ == "__main__":
    main()
