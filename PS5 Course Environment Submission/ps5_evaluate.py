"""Evaluate the two frozen PS5 models using the course Python environment."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import csv
import os

from scipy.special import expit, softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import _libsvm
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd


def prior_counts(times, days):
    """Count earlier events in a trailing window, excluding simultaneous events."""
    return times.searchsorted(times) - times.searchsorted(times - days * 86400)


def fraud_features(events):
    # Return features in input order, including when called on a shuffled batch.
    d = events.reset_index(drop=True).sort_values("unix_time", kind="stable")
    x = pd.DataFrame(index=d.index)
    x["amount"] = d.amt
    x["hour"] = d.trans_time.str.slice(0, 2).astype(int)
    lat1, lat2 = np.radians(d.lat), np.radians(d.merch_lat)
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(np.radians(d.merch_long-d.long)/2)**2
    x["home_merchant_km"] = 12742 * np.arcsin(np.sqrt(a.clip(0, 1)))

    d["time_band"] = x["hour"] // 6
    entities = {"card":["cc_num"], "card_category":["cc_num","category"],
                "account":["acct_num"], "person":["first","last","dob"],
                "card_merchant":["cc_num","merchant"], "category":["category"], "merchant":["merchant"],
                "merchant_time":["merchant","time_band"], "card_time":["cc_num","time_band"]}
    for prefix, keys in entities.items():
        # Collapse simultaneous events so their within-second ordering is immaterial.
        group_keys = keys + ["unix_time"]
        present = d[keys].notna().all(axis=1)
        batch = (d.loc[present].groupby(group_keys, sort=False, observed=True)
                 .amt.agg(total="sum", n="size").reset_index())
        groups = batch.groupby(keys, sort=False, observed=True)
        count = groups.n.cumsum() - batch.n
        total = groups.total.cumsum() - batch.total
        batch["count"] = count
        batch["mean"] = total / count.replace(0, np.nan)
        batch["elapsed"] = groups.unix_time.diff()
        prior = d[group_keys].merge(batch[group_keys + ["count","mean","elapsed"]],
                                    on=group_keys, how="left", sort=False, validate="many_to_one")
        prior.index = d.index
        x[prefix+"_count"] = prior["count"].fillna(0)
        x[prefix+"_amount_ratio"] = d.amt / prior["mean"].replace(0, np.nan)
        if prefix == "card":
            x["card_elapsed"] = prior["elapsed"]
    # Prefix sums and binary search make strict trailing windows inexpensive per card.
    for name in ("hour_count","day_count","day_spend"):
        x[name] = 0.
    x["reference_days_observed"] = 0.
    x["day_vs_prior28_ratio"] = np.nan
    for _, block in d.groupby("cc_num", sort=False):
        times = block.unix_time.to_numpy()
        end = times.searchsorted(times, side="left")
        hour = times.searchsorted(times-3600)
        day = times.searchsorted(times-86400)
        sums = np.r_[0., block.amt.to_numpy().cumsum()]
        x.loc[block.index, ["hour_count","day_count","day_spend"]] = np.column_stack(
            [end-hour, end-day, sums[end]-sums[day]])
        ref_start = times.searchsorted(times-29*86400, side="left")
        ref_total = sums[day]-sums[ref_start]
        coverage = np.clip((times-times[0])/86400-1, 0, 28)
        ratio = np.full(len(block), np.nan)
        np.divide(sums[end]-sums[day], ref_total/28, out=ratio,
                  where=(coverage == 28) & (ref_total > 0))
        x.loc[block.index, ["reference_days_observed","day_vs_prior28_ratio"]] = np.column_stack([coverage,ratio])
    # Count category and overall traffic over aligned historical windows.
    t = d.unix_time.to_numpy()
    c7=np.zeros(len(t),int); c35=c7.copy()
    for idx in d.reset_index(drop=True).groupby("category",sort=False).indices.values():
        c7[idx]=prior_counts(t[idx],7); c35[idx]=prior_counts(t[idx],35)
    n7,n35=prior_counts(t,7),prior_counts(t,35)
    x["recent_category_share"]=(c7+.5)/(n7+7)
    x["category_share_change"]=np.log(x.recent_category_share.to_numpy()/((c35-c7+.5)/(n35-n7+7)))
    return x.sort_index().astype("float32")


def arrays(path):
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def trees(data, prefix, x):
    feature = data[prefix + "feature"]
    left = data[prefix + "left"]
    right = data[prefix + "right"]
    value = data[prefix + "value"]
    cutoff = data[prefix + "cutoff"]
    missing_left = data[prefix + "missing_left"]
    roots = data[prefix + "root"]
    classes = data[prefix + "class"]
    result = np.broadcast_to(data[prefix + "base"], (len(x), len(data[prefix + "base"]))).copy()
    strict = str(data[prefix + "kind"]) == "xgb"
    if strict:
        # The original XGBoost CPU predictor reads numeric features as float32.
        x = np.asarray(x, dtype=np.float32)
    all_rows = np.arange(len(x))
    for root, label in zip(roots, classes):
        nodes = np.full(len(x), root, dtype=left.dtype)
        active = all_rows
        while active.size:
            active = active[left[nodes[active]] >= 0]
            parent = nodes[active]
            observed = x[active, feature[parent]]
            goes_left = observed < cutoff[parent] if strict else observed <= cutoff[parent]
            goes_left = np.where(np.isnan(observed), missing_left[parent], goes_left)
            nodes[active] = np.where(goes_left, left[parent], right[parent])
        result[:, label] += value[nodes]
    objective = str(data[prefix + "objective"])
    if objective == "mean":
        return result / len(roots)
    if objective == "logistic":
        prob = expit(result)
        return np.column_stack((1 - prob[:, 0], prob[:, 0]))
    return softmax(result, axis=1)


class FraudPredictor:
    def __init__(self, path):
        self.data = arrays(path)
        self.fields = self.data["fields"].tolist()
        self.columns = self.data["columns"].tolist()
        self.threshold = float(self.data["threshold"])

    def fraud_probability(self, transactions, history):
        if transactions.empty:
            return np.empty(0)
        # The evaluator always starts a standalone CSV with empty history.
        import pandas as pd
        combined = pd.concat([history[self.fields], transactions[self.fields]], ignore_index=True)
        x = fraud_features(combined).iloc[len(history):][self.columns].to_numpy(dtype=np.float32)
        missing = np.isnan(x)
        x = np.column_stack((np.where(missing, self.data["fill"], x), missing[:, self.data["indicator"]]))
        forest = trees(self.data, "rf_", x)[:, 0]
        boost = trees(self.data, "gb_", x)[:, 1]
        return float(self.data["forest_weight"]) * forest + np.float32(self.data["boost_weight"]) * boost


def scaled(x, mean, scale):
    return (x - mean) / scale


def activity_neighbors(data, prefix, query):
    ref = data[prefix + "refs"]
    keep = data[prefix + "varying"]
    query = scaled(query, data[prefix + "ref_mean"], data[prefix + "ref_scale"])[:, keep]
    distance, near = NearestNeighbors(n_neighbors=15).fit(ref).kneighbors(query)
    weights = 1 / np.maximum(distance, 1e-8)
    return (data[prefix + "votes"][near] * weights[:, :, None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)


def svm(data, prefix, query):
    # Use the course's libsvm implementation with the exported RBF and probability parameters.
    x = np.asarray(scaled(query, data[prefix + "svm_mean"], data[prefix + "svm_scale"]), order="C")
    return _libsvm.predict_proba(
        x, data[prefix + "support"], data[prefix + "support_vectors"], data[prefix + "n_support"],
        data[prefix + "dual"], data[prefix + "intercept"], data[prefix + "prob_a"], data[prefix + "prob_b"],
        kernel="rbf", gamma=float(data[prefix + "gamma"]),
    )


class ActivityPredictor:
    def __init__(self, path):
        self.data = arrays(path)
        self.labels = self.data["labels"]

    def fold_scores(self, i, x):
        data = self.data
        prefix = f"fold{i}_"
        shares = activity_neighbors(data, prefix, x)
        sensors = scaled(x, data[prefix + "sensor_mean"], data[prefix + "sensor_scale"])
        aug = np.column_stack((sensors, scaled(shares, data[prefix + "share_mean"], data[prefix + "share_scale"])))
        score = trees(data, prefix + "general_", aug)
        pair = data[prefix + "pair"]
        mass = score[:, pair].sum(axis=1, keepdims=True)
        refined = (score[:, pair] + mass * trees(data, prefix + "specialist_", aug)) / 2
        score[:, pair] = (refined + trees(data, prefix + "resampled_", sensors)[:, pair]) / 2
        score = score * data[prefix + "frequencies"] ** (-float(data["prior_exponent"]))
        normalized = score / score.sum(axis=1, keepdims=True)
        weight = float(data["svm_weight"])
        return (1 - weight) * normalized + weight * svm(data, prefix, x)

    def predict_scores(self, x):
        with threadpool_limits(limits=1, user_api="blas"):
            return np.mean([self.fold_scores(i, np.asarray(x)) for i in range(int(self.data["fold_count"]))], axis=0)

    def predict(self, x):
        return self.labels[self.predict_scores(x).argmax(axis=1)]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraud-test", type=Path, required=True, help="Headered CSV; last column is is_fraud.")
    parser.add_argument("--activity-test", type=Path, required=True, help="Headerless CSV; 328 sensors then activity label.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def evaluate_fraud(path):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    model = FraudPredictor(Path(__file__).resolve().parent / "best_models" / "fraud.npz")
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

    model = ActivityPredictor(Path(__file__).resolve().parent / "best_models" / "activity.npz")
    labels = model.labels
    data = pd.read_csv(path, header=None)
    if data.shape[1] != 329 or data.empty:
        raise ValueError("Expected activity data with 328 sensor columns and one final activity label.")
    truth = data.iloc[:, -1]
    if not truth.isin(labels).all():
        raise ValueError("Activity CSV contains a missing or unknown gold activity label.")
    pred = model.predict(data.iloc[:, :-1].to_numpy(dtype=float))
    return pred, {
        "Accuracy": accuracy_score(truth, pred),
        "Macro F1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
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
    run_evaluation(args.fraud_test, args.activity_test, args.output_dir)


def run_evaluation(fraud_test, activity_test, output_dir):
    fraud_test, activity_test, output_dir = map(Path, (fraud_test, activity_test, output_dir))
    tasks = [
        ("Fraud", fraud_test, "fraud_predictions.csv", evaluate_fraud, True),
        ("Activity", activity_test, "activity_predictions.csv", evaluate_activity, False),
    ]
    inputs = {fraud_test.resolve(), activity_test.resolve()}
    if any((output_dir / name).resolve() in inputs for _, _, name, _, _ in tasks):
        raise ValueError("An output filename would overwrite a test input. Choose another output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Publish the output files after both evaluations and CSV writes succeed.
    reports = []
    with TemporaryDirectory(dir=output_dir) as tmp:
        for task, path, name, evaluate, header in tasks:
            pred, metrics = evaluate(path)
            write_predictions(path, Path(tmp) / name, pred, header)
            reports.append((task, len(pred), metrics, name))
        for _, _, _, name in reports:
            os.replace(Path(tmp) / name, output_dir / name)
    for task, count, metrics, name in reports:
        print(f"\n{task} ({count:,} rows)")
        for metric, value in metrics.items():
            result = "undefined (only one gold class)" if value is None else f"{value:.6f}"
            print(f"  {metric}: {result}")
        print(f"  Predictions: {output_dir / name}")


if __name__ == "__main__":
    main()
