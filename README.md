# VLMGuard

Implementation of **VLMGuard: Bootstrapping Malicious Prompt Detectors from Unlabeled Vision-Language Prompts in the Wild** ([TMLR 2026](https://openreview.net/forum?id=z7gczmhmmo), [arXiv](https://arxiv.org/abs/2410.00296)).

VLMGuard estimates maliciousness from unlabeled vision-language prompts and trains a prompt detector using the resulting pseudo-labels. This repository provides the data-preparation, feature-extraction, training, evaluation, and inference workflows. The verified JailBreakV/GPT4V detector is available on [Hugging Face](https://huggingface.co/Sol45/VLMGuard-Qwen2.5-VL-7B-JailBreakV-GPT4V). See [Models](docs/MODELS.md) for its format and usage.

## Requirements

Python 3.10+ and a CUDA-capable environment are recommended for feature extraction. Install dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

Training from cached NumPy features can also run without a GPU. All paths are command-line arguments; there are no dataset-name switches or benchmark-specific defaults.

## Prepare Data

Each split is a JSON list. Rows need a stable `id`, or both `image_path` and `question`; validation and test rows also need a binary `label` (`0` benign, `1` malicious). Feature arrays must preserve manifest row order. Train, validation, and test must be disjoint. Training labels are not used by the method.

Use existing disjoint manifests for an exact reproduction. To construct a new split from four source pools:

```bash
python prepare_splits.py \
  --train-benign data/source/train_benign.json \
  --train-malicious data/source/train_malicious.json \
  --test-benign data/source/test_benign.json \
  --test-malicious data/source/test_malicious.json \
  --validation-size 100 --validation-positives 1 \
  --malicious-ratio 0.005 --seed 777 \
  --output-dir data/splits/run777
```

These numbers are examples, not paper-mandated settings. See [Dataset Preparation](docs/DATASETS.md) for benchmark sources, reported counts, and exact-reproduction caveats.

## Extract Features

Run extraction on a machine with a CUDA GPU, once per manifest:

```bash
python extract_features.py \
  --manifest data/splits/run777/train.json \
  --image-root data/images \
  --model models/Qwen2.5-VL-7B-Instruct \
  --output features/run777/train.npy
```

Repeat for validation and test. `--start` and `--stop` support independent shards; concatenate shard arrays in manifest order. The extractor records the final input token at each Qwen2.5-VL block output without generating answers. Other backbones may supply aligned `(N, L, H)` arrays through the same interface.

## Train And Evaluate

```bash
python train.py \
  --train-manifest data/splits/run777/train.json \
  --train-features features/run777/train.npy \
  --validation-manifest data/splits/run777/validation.json \
  --validation-features features/run777/validation.npy \
  --test-manifest data/splits/run777/test.json \
  --test-features features/run777/test.npy \
  --output-dir results/run777
```

Use `--layers all`, `--layers 3:9`, or `--layers 3,5,8`. `--min-k` and `--max-k` are inclusive; the default searches 1 through 5. The run writes `classifier.pt`, `subspace.npz`, and `metrics.json` with classifier and direct-projection test metrics.

The default pipeline uses unweighted SVD pseudo-labels and a two-linear-layer classifier; `--weighted-svd` enables singular-value weighting. Training defaults are 30 epochs, Adam, learning rate 0.001, batch size 64, hidden width 256, no weight decay, and seed 2001. [Method](docs/METHOD.md) describes the stages and configurable choices.

## Inference

```bash
python infer.py \
  --run-dir results/run777 \
  --features features/new.npy \
  --output results/new_scores.npy
```

The output is a one-dimensional NumPy array of maliciousness probabilities in input row order.

## Tests

```bash
python -m pytest -q
```

## Citation

If you use VLMGuard, please cite the TMLR paper:

```bibtex
@article{fang2026vlmguard,
  title={VLMGuard: Bootstrapping Malicious Prompt Detectors from Unlabeled Vision-Language Prompts in the Wild},
  author={Fang, Junlin and Chen, Wenyu and Ghosh, Reshmi and Sim, Robert and Salem, Ahmed and Carvalho, Vitor R. and Lawton, Emily and Li, Sharon and Stokes, Jack W. and Du, Sean},
  journal={Transactions on Machine Learning Research},
  year={2026},
  url={https://openreview.net/forum?id=z7gczmhmmo}
}
```

The same metadata is available in [`CITATION.cff`](CITATION.cff).
