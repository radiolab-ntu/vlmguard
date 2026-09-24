# Dataset Preparation

No datasets, images, or precomputed features are distributed in this repository. Download and normalize source material separately under its own terms. Manifests and feature paths are supplied at run time, with no dataset-specific code paths.

## Manifest Contract

Each split is an external JSON list. Each row has a stable `id`, or both `image_path` and `question`. Validation and test rows additionally have `label` (`0` benign, `1` malicious). For Qwen extraction, every row needs `image_path` and `question`; relative image paths resolve against `--image-root`. Feature arrays must follow the exact manifest row order. The generic split builder removes source labels from its training output.

Preserve the benchmark's published train/test division. Hold out validation examples before sampling an unlabeled training mixture. For exact reproduction, use the original held-out validation IDs rather than silently regenerating a split.

## Paper Benchmarks

The VLMGuard paper's Appendix A reports these counts. They describe benchmark construction; none are constants in this code:

| Scenario | Full train benign/malicious | Main-mixture benign/malicious | Test benign/malicious |
| --- | ---: | ---: | ---: |
| JailBreakV & GPT4V | 16,023 / 22,377 | 16,023 / 80 | 3,977 / 5,623 |
| VLGuard & MLLMGuard | 977 / 1,603 | 977 / 4 | 558 / 587 |
| VLGuard & MSSBench | 1,275 / 1,326 | 1,275 / 6 | 636 / 515 |

The paper uses the official VLGuard split and an 80/20 split for JailBreakV-28K, GPT4V-Caption, MLLMGuard, and MSSBench. Original full dataset sizes reported in the appendix include 2,282 MLLMGuard malicious image-text pairs. MLLMGuard's public release describes a 1,500-sample sanitized subset; it is not interchangeable with that full collection. Verify access to the full assets and record source revisions before claiming exact reproduction.

Sources: [JailBreakV-28K](https://github.com/SaFoLab-WISC/JailBreakV_28K), [GPT4V-Caption](https://huggingface.co/datasets/laion/gpt4v-dataset), [VLGuard](https://github.com/ys-zong/VLGuard), [MLLMGuard](https://github.com/AI45Lab/MLLMGuard), and [MSSBench](https://github.com/UCSB-AI/MSSBench).
