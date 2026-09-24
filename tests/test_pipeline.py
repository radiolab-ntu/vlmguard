import json
import sys

import numpy as np

from vlmguard.infer import main as infer_main
from vlmguard.train import main as train_main
from verify_model import verify_run


def test_train_save_and_infer_round_trip(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    labels = np.array([0] * 10 + [1] * 10)
    train = rng.normal(size=(80, 1, 4)).astype(np.float32)
    train[-5:, 0, 0] += 5
    validation = rng.normal(size=(20, 1, 4)).astype(np.float32)
    test = rng.normal(size=(20, 1, 4)).astype(np.float32)
    validation[labels == 1, 0, 0] += 5
    test[labels == 1, 0, 0] += 5
    for name, features, prefix, labeled in (
        ("train", train, "tr", False),
        ("validation", validation, "va", True),
        ("test", test, "te", True),
    ):
        np.save(tmp_path / f"{name}.npy", features)
        rows = [
            {"id": f"{prefix}{i}", **({"label": int(labels[i])} if labeled else {})}
            for i in range(len(features))
        ]
        (tmp_path / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "run"
    arguments = ["vlmguard.train"]
    for name in ("train", "validation", "test"):
        arguments.extend([f"--{name}-manifest", str(tmp_path / f"{name}.json")])
        arguments.extend([f"--{name}-features", str(tmp_path / f"{name}.npy")])
    arguments.extend(["--output-dir", str(output), "--device", "cpu", "--min-k", "1", "--max-k", "1", "--epochs", "1", "--batch-size", "16", "--hidden-dim", "8"])
    monkeypatch.setattr(sys, "argv", arguments)
    train_main()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert 0 <= metrics["test_auroc"] <= 1
    release = verify_run(output)
    assert release["test_auroc"] == metrics["test_auroc"]
    assert len(release["sha256"]["classifier.pt"]) == 64
    score_path = tmp_path / "scores.npy"
    monkeypatch.setattr(sys, "argv", ["vlmguard.infer", "--run-dir", str(output), "--features", str(tmp_path / "test.npy"), "--output", str(score_path), "--device", "cpu"])
    infer_main()
    scores = np.load(score_path)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
