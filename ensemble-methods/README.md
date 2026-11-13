# Ensemble Methods Case Study

Group project. Compares a single decision tree with bagging and random forests on two classification problems, tuning each model with grid search cross-validation.

[Ensemble_Methods_Case_Study.ipynb](Ensemble_Methods_Case_Study.ipynb)

## 1. Online news popularity (Mashable)
Predict whether an article gets more shares than the median.

| Model | Accuracy | ROC-AUC | MSPE |
|---|---|---|---|
| Decision Tree | 0.622 | 0.669 | 0.228 |
| Bagging | 0.645 | 0.698 | 0.221 |
| Random Forest | 0.642 | 0.699 | 0.220 |

Bagging had the highest accuracy, while the random forest had the best ROC-AUC and the lowest MSPE, meaning its predicted probabilities were the most accurate.

## 2. Spam email detection
Built 33 binary features from raw emails (spam keywords, links/HTML, dollar signs, all-caps words) and compared the ensembles with a Naive Bayes baseline.

| Model | Accuracy |
|---|---|
| Naive Bayes | 0.902 |
| Decision Tree | 0.948 |
| Bagging | 0.968 |
| Random Forest | 0.968 |

The ensembles outperformed Naive Bayes. The most important feature was whether an email contains an HTML link.

## Data
- `MashableData/`: Online News Popularity training and test sets (UCI Machine Learning Repository). The notebook loads them from this folder, so run it from `ensemble-methods/`.
- Spam email corpus: about 4,300 labelled `.eml` files plus a label file. They're not included here because of the file count and size, so the spam section still loads them from the original Google Drive paths.
