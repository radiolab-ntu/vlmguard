import json

import numpy as np
import pytest

from vlmguard.data import assert_disjoint, load_features, load_manifest


def test_manifest_requires_binary_labels(tmp_path):
    path = tmp_path / "validation.json"
    path.write_text(json.dumps([{"id": "a", "label": 2}]), encoding="utf-8")
    with pytest.raises(ValueError, match="binary label"):
        load_manifest(str(path), require_labels=True)


def test_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([{"id": "same"}], [{"id": "same"}])


def test_feature_rows_are_checked(tmp_path):
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((2, 3), dtype=np.float32))
    assert load_features(str(path), 2).shape == (2, 1, 3)
    with pytest.raises(ValueError, match="does not match"):
        load_features(str(path), 3)
