# PS5 team system — fraud and activity recognition

## Evaluate in Jupyter or Colab
Open `Evaluate Test Sets.ipynb` from the extracted folder. Place both labeled test CSVs beside it, enter their filenames in the first code cell, and choose **Run all**. For Colab, open the notebook and upload the original submission ZIP and both CSVs to the runtime's Files panel. The notebook runs the packaged evaluator, prints all six metrics and writes both prediction CSVs plus `metrics.txt` under `results/`.

A Python environment is the interpreter and library versions running a program. The notebook checks those versions automatically. If setup is needed, it uses uv (bootstrapped with pip using uv==0.13.9 if absent) to run the evaluator in a separate Python 3.12 environment. First setup requires permission and network access for downloads or a populated cache. In a restricted lab, use the preinstalled matching environment or provision the requirements before evaluation. The steps below are the alternative for command-line grading.

### If the notebook stops
- A Jupyter or VS Code prompt for ipykernel appears before any cell runs. Select an existing Jupyter kernel with ipykernel installed (or use a Colab hosted runtime).
- A missing CSV message lists the CSV files visible in the working directory. Enter the two actual filenames in the first code cell and run it again. Use the released test data for the report.
- If model setup or prediction fails, read the evaluator error printed above the final exception. The notebook prints the kernel's Python version and the requirements that differ. An initial isolated setup needs a package cache or network access.

## Run the frozen system
Extract the complete ZIP. From its extracted directory, in an activated Python 3.12 environment:

```sh
python -m pip install -r requirements.txt
python evaluate.py --fraud-test "/path/to/fraud_test.csv" --activity-test "/path/to/activity_test.csv" --output-dir results
```

The paths may be absolute or relative to the working directory. The evaluator finds its two trusted model files in `best_models/` beside the script. The export environment used Python 3.12.14; the requirements record the modeling stack, including imbalanced-learn for the training notebook. On the course server, check the Python version and install missing or mismatched requirements in a writable environment before running. Full evaluation uses CPU.

## Inputs and metrics
- **Fraud:** Supply the original headered CSV with `is_fraud` as its last column (0 = legitimate, 1 = fraud). The forest uses transaction history available strictly before each timestamp in this test batch. The first transactions start with empty history. Pass the whole file in a single invocation. Test labels are used solely for metrics. The script prints accuracy, two-class macro F1, fraud F1 and ROC AUC. ROC AUC uses continuous positive-class probabilities; class predictions use the frozen development threshold. A single-class file has undefined ROC AUC.
- **Activity:** Supply a headerless CSV with 328 numeric sensor columns in training order and the gold activity name last. The four frozen fold models use their stored neighbor references and produce one ensemble decision per reading. The script prints accuracy and macro F1 across the 11 trained labels. Both tasks assign zero F1 to a class with no true or predicted examples.

## Outputs
`results/fraud_predictions.csv` and `results/activity_predictions.csv` retain the input row order, header convention and original feature field values. The final column contains predicted labels. A successful rerun replaces these two output files. Save terminal output for the report. The previously reported development scores used separate folds/periods; this command prints scores for the files actually supplied.

## Notebook and model files
`PS5 Best Models.ipynb` documents the seed, training, development evaluation and export for both tasks. It is included for reproducibility; run only the standalone evaluator for the released test set. Rerunning the notebook's training/export cells from the bundle directory replaces its model files, so use a separate working copy for training. Training data and the released test files are supplied separately by the course.
