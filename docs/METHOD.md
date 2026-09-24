# Method

1. Extract each VLM block's final input-token representation. `extract_features.py` supports Qwen2.5-VL. Other backbones can supply aligned `(N, L, H)` NumPy arrays; a single-layer `(N, H)` array is also accepted for training.
2. Fit centered SVD on the unlabeled training mixture. By default, score each prompt with `mean_j(centered_projection_j**2)`; `--weighted-svd` applies singular-value weighting.
3. Select layer and `k` using projection-score AUROC on a disjoint labeled validation set. Select the projection-score threshold using validation balanced accuracy, then pseudo-label the unlabeled training mixture.
4. Fit a two-linear-layer ReLU classifier on the selected layer's last-token features. Select the checkpoint by validation AUROC and evaluate it on the held-out test set.

The current defaults use hidden width 256, 30 epochs, Adam, learning rate 0.001, cosine decay, batch size 64, no weight decay, and class-weighted cross-entropy. Data and feature paths remain caller-provided, with no benchmark-specific switches.
