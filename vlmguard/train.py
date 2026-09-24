"""Train VLMGuard from external last-token feature arrays and JSON manifests."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score

from vlmguard.data import assert_disjoint, load_features, load_manifest
from vlmguard.scoring import score, select_subspace, select_threshold


def parse_layers(text: str, count: int) -> list[int]:
    if text == "all":
        return list(range(count))
    if ":" in text:
        start, stop = (int(part) for part in text.split(":"))
        layers = list(range(start, stop))
    else:
        layers = [int(part) for part in text.split(",")]
    if not layers or len(layers) != len(set(layers)) or any(i < 0 or i >= count for i in layers):
        raise ValueError("invalid layer selection")
    return layers


class DetectorClassifier(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, features):
        return self.fc3(self.dropout(torch.relu(self.fc1(features))))


def make_classifier(dimension: int, hidden: int, dropout: float = 0.0):
    return DetectorClassifier(dimension, hidden, dropout)


def train_classifier(features, pseudo_labels, validation_features, validation_labels, args):
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model = make_classifier(features.shape[1], args.hidden_dim, args.dropout).to(device)
    x = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
    y = torch.from_numpy(np.asarray(pseudo_labels, dtype=np.int64))
    counts = np.bincount(pseudo_labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("both pseudo-label classes are required")
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)
    class_weights = counts.sum() / (2 * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_auc = -float("inf")
    best_state = None
    for _ in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_scores = predict(model, validation_features, device, args.batch_size)
        val_auc = float(roc_auc_score(validation_labels, val_scores))
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc


def predict(model, features, device, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.array(features[start:start + batch_size], dtype=np.float32, copy=True)).to(device)
            chunks.append(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}-manifest", required=True)
        parser.add_argument(f"--{split}-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all", help="all, 3:9 (exclusive end), or 3,5,8")
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--threshold-steps", type=int, default=40)
    parser.add_argument("--weighted-svd", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_dim < 1 or not 0 <= args.dropout < 1:
        parser.error("epochs, batch-size, and hidden-dim must be positive")
    manifests = [load_manifest(getattr(args, f"{part}_manifest"), part != "train") for part in ("train", "validation", "test")]
    assert_disjoint(*manifests)
    arrays = [load_features(getattr(args, f"{part}_features"), len(rows)) for part, rows in zip(("train", "validation", "test"), manifests)]
    train, validation, test = arrays
    if train.shape[1:] != validation.shape[1:] or train.shape[1:] != test.shape[1:]:
        raise ValueError("all feature arrays must have the same layer and hidden dimensions")
    val_labels = np.array([int(row["label"]) for row in manifests[1]], dtype=np.int64)
    test_labels = np.array([int(row["label"]) for row in manifests[2]], dtype=np.int64)
    if set(test_labels) != {0, 1}:
        raise ValueError("test manifest must contain both classes")
    layers = parse_layers(args.layers, train.shape[1])
    selected, selection_auc = select_subspace(train, validation, val_labels, layers, args.min_k, args.max_k, weighted=args.weighted_svd)
    train_scores = score(train[:, selected.layer], selected, weighted=args.weighted_svd)
    val_scores = score(validation[:, selected.layer], selected, weighted=args.weighted_svd)
    threshold, validation_balanced_accuracy, pseudo_fraction = select_threshold(
        train_scores, val_scores, val_labels, args.threshold_steps
    )
    pseudo_labels = (train_scores > threshold).astype(np.int64)
    classifier, classifier_val_auc = train_classifier(
        train[:, selected.layer], pseudo_labels, validation[:, selected.layer], val_labels, args
    )
    test_scores = predict(classifier, test[:, selected.layer], torch.device(args.device), args.batch_size)
    direct_test_scores = score(test[:, selected.layer], selected, weighted=args.weighted_svd)
    metrics = {
        "test_auroc": float(roc_auc_score(test_labels, test_scores)),
        "test_aupr": float(average_precision_score(test_labels, test_scores)),
        "direct_projection_test_auroc": float(roc_auc_score(test_labels, direct_test_scores)),
        "direct_projection_test_aupr": float(average_precision_score(test_labels, direct_test_scores)),
        "validation_projection_auroc": selection_auc,
        "validation_threshold_balanced_accuracy": validation_balanced_accuracy,
        "validation_classifier_auroc": classifier_val_auc,
        "layer": selected.layer,
        "k": selected.k,
        "threshold": threshold,
        "pseudo_positive_fraction": pseudo_fraction,
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "config": vars(args),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.cpu().state_dict(), output / "classifier.pt")
    np.savez(output / "subspace.npz", mean=selected.mean, vectors=selected.vectors, singular_values=selected.singular_values, layer=selected.layer, k=selected.k, threshold=threshold, weighted=int(args.weighted_svd))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ("test_auroc", "test_aupr", "direct_projection_test_auroc", "layer", "k", "pseudo_positive_fraction")}, indent=2))


if __name__ == "__main__":
    main()
