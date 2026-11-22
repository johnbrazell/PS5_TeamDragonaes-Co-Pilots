"""Evaluate both frozen PS5 classifiers on labeled test CSVs."""
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraud-test", type=Path, required=True, help="Headered CSV; last column is is_fraud.")
    parser.add_argument("--activity-test", type=Path, required=True, help="Headerless CSV; 328 sensors then activity label.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def load_model(filename):
    path = Path(__file__).resolve().parent / "best_models" / filename
    with path.open("rb") as stream:
        return pickle.load(stream)


def evaluate_fraud(path):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model = load_model("fraud_best_random_forest_transaction_history.pkl")
    header = pd.read_csv(path, nrows=0).columns
    if header[-1] != "is_fraud":
        raise ValueError("Fraud CSV must have is_fraud as its final column.")
    ids = ["cc_num", "acct_num", "first", "last", "dob", "merchant", "category"]
    data = pd.read_csv(path, usecols=[*model.fields, "is_fraud"], dtype={k: "string" for k in ids})
    truth = data.pop("is_fraud")
    if truth.empty or not truth.isin([0, 1]).all():
        raise ValueError("Expected fraud labels of 0 or 1 on every row.")
    # The frozen feature builder sorts internally and excludes the current timestamp.
    # Start without earlier training transactions; preserve input order in predictions.
    scores = model.fraud_probability(data, data.iloc[:0])
    if not np.isfinite(scores).all():
        raise ValueError("The fraud model returned a nonfinite score.")
    pred = (scores >= model.threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(truth, pred),
        "Macro F1": f1_score(truth, pred, labels=[0, 1], average="macro", zero_division=0),
        "Fraud F1": f1_score(truth, pred, pos_label=1, zero_division=0),
        "ROC AUC": roc_auc_score(truth, scores) if truth.nunique() == 2 else None,
    }
    return pred, metrics


def evaluate_activity(path):
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score

    model = load_model("activity_best_15_neighbor_sports_booster_ensemble.pkl")
    data = pd.read_csv(path, header=None)
    if data.shape[1] != 329 or data.empty:
        raise ValueError("Expected activity data with 328 sensor columns and one final activity label.")
    truth = data.iloc[:, -1]
    if not truth.isin(model.labels).all():
        raise ValueError("Activity CSV contains a missing or unknown gold activity label.")
    pred = model.predict(data.iloc[:, :-1].to_numpy(dtype=float))
    return pred, {
        "Accuracy": accuracy_score(truth, pred),
        "Macro F1": f1_score(truth, pred, labels=model.labels, average="macro", zero_division=0),
    }


def write_predictions(source, destination, predictions, has_header):
    """Copy original field strings, including identifiers and numeric formatting."""
    with source.open(encoding="utf-8-sig", newline="") as inp, destination.open("w", encoding="utf-8", newline="") as out:
        rows, writer = csv.reader(inp), csv.writer(out)
        if has_header:
            writer.writerow(next(rows))
        for row, prediction in zip((row for row in rows if row), predictions, strict=True):
            row[-1] = prediction
            writer.writerow(row)


def main():
    args = arguments()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use a Python 3.12 environment for these frozen cloudpickle models.")
    tasks = [
        ("Fraud", args.fraud_test, "fraud_predictions.csv", evaluate_fraud, True),
        ("Activity", args.activity_test, "activity_predictions.csv", evaluate_activity, False),
    ]
    inputs = {args.fraud_test.resolve(), args.activity_test.resolve()}
    if any((args.output_dir / name).resolve() in inputs for _, _, name, _, _ in tasks):
        raise ValueError("An output filename would overwrite a test input. Choose another output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Publish the output files after both evaluations and CSV writes succeed.
    reports = []
    with TemporaryDirectory(dir=args.output_dir) as tmp:
        for task, path, name, evaluate, header in tasks:
            pred, metrics = evaluate(path)
            write_predictions(path, Path(tmp) / name, pred, header)
            reports.append((task, len(pred), metrics, name))
        for _, _, _, name in reports:
            os.replace(Path(tmp) / name, args.output_dir / name)
    for task, count, metrics, name in reports:
        print(f"\n{task} ({count:,} rows)")
        for metric, value in metrics.items():
            result = "undefined (only one gold class)" if value is None else f"{value:.6f}"
            print(f"  {metric}: {result}")
        print(f"  Predictions: {args.output_dir / name}")


if __name__ == "__main__":
    main()
