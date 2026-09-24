"""Dataset-neutral manifest validation and feature loading."""

import json
from pathlib import Path

import numpy as np


def load_manifest(path: str, require_labels: bool = False) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest must be a nonempty JSON list: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest contains a non-object row: {path}")
    if require_labels and any(row.get("label") not in (0, 1, False, True) for row in rows):
        raise ValueError(f"labeled manifest needs binary label fields: {path}")
    return rows


def sample_id(row: dict) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    if row.get("image_path") is not None and row.get("question") is not None:
        return json.dumps([row["image_path"], row["question"]], ensure_ascii=False)
    raise ValueError("each manifest row needs id or image_path and question for overlap checks")


def assert_disjoint(*manifests: list[dict]) -> None:
    seen: set[str] = set()
    for manifest in manifests:
        ids = [sample_id(row) for row in manifest]
        current = set(ids)
        if len(current) != len(ids):
            raise ValueError("manifest contains duplicate sample IDs")
        if seen & current:
            raise ValueError("train, validation, and test manifests overlap")
        seen.update(current)


def load_features(path: str, expected_rows: int) -> np.ndarray:
    features = np.load(path, mmap_mode="r")
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or features.shape[0] != expected_rows:
        raise ValueError(f"feature shape {features.shape} does not match {expected_rows} manifest rows")
    return features
