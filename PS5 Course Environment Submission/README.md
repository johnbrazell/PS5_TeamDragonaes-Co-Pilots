# PS5 course environment submission

This bundle runs on the `idai610` course kernel. It uses NumPy, pandas, SciPy, scikit-learn and threadpoolctl, an installed dependency of scikit-learn. The model exports in `best_models/` are fitted parameter arrays loaded with `allow_pickle=False`. The notebook uses the exported parameters and fixed decision rules for inference.

## Run in Jupyter

1. Extract the entire ZIP into one folder and keep the `best_models` subfolder intact.
2. Place the competition CSVs next to `Evaluate Test Sets.ipynb`. Use `fraud_test.csv` and `activity_test.csv`, or change the paths in the first cell.
3. Open the notebook, select the `idai610` kernel and choose **Run All**.
4. Collect the two prediction CSVs and `metrics.txt` from the `results` folder.

## Run from the command line

From the extracted folder in the activated `idai610` environment:

```sh
python ps5_evaluate.py --fraud-test fraud_test.csv --activity-test activity_test.csv --output-dir results
```

Python callers can import `evaluate_fraud(path)` and `evaluate_activity(path)` from `ps5_evaluate`. Each returns predictions and a dictionary of metric values. The notebook and CLI use the same functions. The CLI prints metrics to the terminal. The notebook also saves them to `results/metrics.txt`.

## Inputs and results

- **Fraud.** Use the headered competition file with `is_fraud` as the final column. Pass the complete file in one run. The predictor starts each file with empty transaction history and uses transactions strictly earlier than each scored purchase. Metrics are accuracy, macro-F1, fraud F1 and ROC AUC.
- **Activity.** Use the headerless file with 328 numeric sensor columns in training order and the gold activity label last. Metrics are accuracy and macro-F1 over the 11 activity labels.
- **Prediction CSVs.** Each file preserves its original row order and input fields. The final column contains the predicted label. Gold labels are used for scoring.
