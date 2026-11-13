# Heart Failure Prediction After Myocardial Infarction

Predicts which heart attack patients go on to develop heart failure, using clinical data recorded at admission.

[heart_failure.py](heart_failure.py)

## Approach
- Renamed the dataset's coded column names (e.g. `ZSN_A`, `K_SH_POST`) to readable clinical terms
- Dropped columns with heavy missingness and other outcome columns that would leak the answer
- Imputed missing values (median for numeric features) and used a stratified 80/20 train/test split
- Trained and compared three classifiers, using class weighting to handle the imbalanced outcome:
  - Logistic regression (standardized features)
  - Random forest (200 trees)
  - K-nearest neighbors (k = 5)
- Evaluated with accuracy, ROC-AUC and confusion matrices

## Output
The script prints a summary of key variables and each model's metrics, and saves 11 charts, including model comparisons, the top 15 random forest feature importances, heart failure rates by age, sex and prior history, complication rates, and a correlation heatmap.

## Data
Myocardial Infarction Complications dataset from the UCI Machine Learning Repository (1,700 patients), included here as `Myocardial_infarction_complications_Database.csv`. From this folder, run:

```bash
python heart_failure.py
```
