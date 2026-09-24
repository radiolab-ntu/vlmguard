# Models

The verified JailBreakV/GPT4V detector is available on [Hugging Face](https://huggingface.co/Sol45/VLMGuard-Qwen2.5-VL-7B-JailBreakV-GPT4V). The Qwen2.5-VL backbone is obtained separately from its model provider; detector checkpoints do not contain backbone weights.

A release artifact for the current inference interface consists of three files in one directory:

| File | Purpose |
| --- | --- |
| `classifier.pt` | Detector classifier weights |
| `subspace.npz` | Selected layer, subspace, and threshold |
| `metrics.json` | Training configuration and evaluation metrics |

The released checkpoint was converted from the verified training run and passed `python verify_model.py --run-dir /path/to/run`. CPU inference on all 9,600 held-out feature rows reproduced AUROC `0.9817013616071427`. The selected feature layer is 8, subspace rank 1, classifier hidden width 256, and backbone revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.

The model repository includes `classifier.pt`, `subspace.npz`, `metrics.json`, a model card and SHA-256 checksums. Its checkpoint was trained with model seed 2001 and split seed 777; held-out AUROC is `0.9817013616071427` and AUPR is `0.9788129385786769`.

Download the files into one directory and run:

```bash
python verify_model.py --run-dir /path/to/model
python infer.py --run-dir /path/to/model --features /path/to/features.npy --output scores.npy --device cpu
```

`infer.py` accepts features with the same hidden dimension and layer indexing as the checkpoint and emits maliciousness probabilities.
