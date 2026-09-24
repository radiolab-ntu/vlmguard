import numpy as np
import torch

from vlmguard.train import make_classifier, predict


def test_classifier_checkpoint_layout_and_prediction():
    model = make_classifier(4, 8)
    assert set(model.state_dict()) == {
        "fc1.weight",
        "fc1.bias",
        "fc3.weight",
        "fc3.bias",
    }
    scores = predict(model, np.zeros((3, 4), dtype=np.float32), torch.device("cpu"), 2)
    assert scores.shape == (3,)
    assert np.all((scores >= 0) & (scores <= 1))
