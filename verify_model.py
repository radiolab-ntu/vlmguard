"""Validate a VLMGuard detector artifact before distribution."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vlmguard.train import make_classifier, predict


FILES = ("classifier.pt", "subspace.npz", "metrics.json")


def verify_run(run_dir):
    run = Path(run_dir)
    paths = {name: run / name for name in FILES}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = json.loads(paths["metrics.json"].read_text(encoding="utf-8"))
    hidden_dim = int(metrics["config"]["hidden_dim"])
    with np.load(paths["subspace.npz"], allow_pickle=False) as subspace:
        dimension = len(subspace["mean"])
        layer = int(subspace["layer"])
        k = int(subspace["k"])
        vectors = subspace["vectors"]
        if dimension < 1 or layer < 0 or k < 1 or vectors.shape != (dimension, k):
            raise ValueError("invalid subspace dimensions or selected layer")
        if subspace["singular_values"].shape != (k,):
            raise ValueError("invalid singular-value dimensions")
        if not all(np.isfinite(subspace[key]).all() for key in ("mean", "vectors", "singular_values")):
            raise ValueError("subspace contains non-finite values")
    if hidden_dim < 1 or metrics["layer"] != layer or metrics["k"] != k:
        raise ValueError("metrics and subspace disagree")

    model = make_classifier(dimension, hidden_dim, float(metrics["config"].get("dropout", 0.0)))
    model.load_state_dict(torch.load(paths["classifier.pt"], map_location="cpu", weights_only=True))
    score = predict(model, np.zeros((1, dimension), dtype=np.float32), torch.device("cpu"), 1)
    if score.shape != (1,) or not np.isfinite(score).all():
        raise ValueError("classifier failed a CPU inference smoke test")

    checksums = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checksums[name] = digest.hexdigest()
    return {
        "layer": layer,
        "k": k,
        "hidden_dim": hidden_dim,
        "feature_dimension": dimension,
        "test_auroc": metrics["test_auroc"],
        "test_aupr": metrics["test_aupr"],
        "sha256": checksums,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
