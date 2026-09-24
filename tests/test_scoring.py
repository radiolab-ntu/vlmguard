import numpy as np
import pytest

from vlmguard.scoring import fit_subspace, score, select_subspace, select_threshold


def test_weighted_projection_uses_unlabeled_mean_and_singular_values():
    train = np.array([[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    fitted = fit_subspace(train, 1)
    np.testing.assert_allclose(fitted.mean, [0.0, 0.0])
    np.testing.assert_allclose(fitted.singular_values, [np.sqrt(20)])
    np.testing.assert_allclose(score(np.array([[2.0, 0.0]]), fitted), [4])
    np.testing.assert_allclose(score(np.array([[2.0, 0.0]]), fitted, weighted=True), [4 * np.sqrt(20)])


def test_layer_and_k_are_selected_from_validation_without_refitting():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(60, 2, 3))
    train[:, 1, 0] *= 5
    val = rng.normal(size=(12, 2, 3))
    labels = np.array([0] * 6 + [1] * 6)
    val[:6, 1, 0] = 0.0
    val[6:, 1, 0] = 8.0
    chosen, auc = select_subspace(train, val, labels, [0, 1], 1, 1)
    assert chosen.layer == 1
    assert auc > 0.9
    np.testing.assert_allclose(chosen.mean, train[:, 1].mean(axis=0))


def test_threshold_requires_two_pseudo_classes():
    with pytest.raises(ValueError, match="single pseudo-label"):
        select_threshold(np.ones(10), np.array([0.0, 1.0]), np.array([0, 1]))
