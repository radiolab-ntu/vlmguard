"""Build generic disjoint manifests from caller-provided benign/malicious pools."""

import argparse
import json
from pathlib import Path

import numpy as np

from vlmguard.data import assert_disjoint, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-benign", required=True)
    parser.add_argument("--train-malicious", required=True)
    parser.add_argument("--test-benign", required=True)
    parser.add_argument("--test-malicious", required=True)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--validation-positives", type=int, default=1)
    parser.add_argument("--malicious-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0 < args.malicious_ratio < 1:
        parser.error("malicious-ratio must be between 0 and 1")
    if not 0 < args.validation_positives < args.validation_size:
        parser.error("validation needs both classes")
    groups = [load_manifest(path) for path in (args.train_benign, args.train_malicious, args.test_benign, args.test_malicious)]
    assert_disjoint(*groups)
    rng = np.random.default_rng(args.seed)

    def shuffled(rows):
        return [rows[int(i)] for i in rng.permutation(len(rows))]

    train_benign, train_malicious, test_benign, test_malicious = map(shuffled, groups)
    neg_count = args.validation_size - args.validation_positives
    if len(train_benign) <= neg_count or len(train_malicious) <= args.validation_positives:
        parser.error("training pools are too small for the requested validation split")
    validation = [{**row, "label": 0} for row in train_benign[:neg_count]] + [
        {**row, "label": 1} for row in train_malicious[:args.validation_positives]
    ]
    remaining_benign = train_benign[neg_count:]
    remaining_malicious = train_malicious[args.validation_positives:]
    malicious_count = round(len(remaining_benign) * args.malicious_ratio / (1 - args.malicious_ratio))
    if not 1 <= malicious_count <= len(remaining_malicious):
        parser.error("requested ratio needs more malicious samples than available")
    mixture = [{key: value for key, value in row.items() if key != "label"} for row in remaining_benign + remaining_malicious[:malicious_count]]
    test = [{**row, "label": 0} for row in test_benign] + [{**row, "label": 1} for row in test_malicious]
    mixture, validation, test = map(shuffled, (mixture, validation, test))
    assert_disjoint(mixture, validation, test)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", mixture), ("validation", validation), ("test", test)):
        (output / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": len(mixture), "validation": len(validation), "test": len(test), "train_malicious": malicious_count}))


if __name__ == "__main__":
    main()
