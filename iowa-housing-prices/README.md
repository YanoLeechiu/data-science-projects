# Iowa Housing Prices

Group project. Uses the Ames, Iowa housing dataset (1,460 home sales, 79 features) to explain what drives sale price and then to predict it.

## Part I: Explanatory modeling
[Part1_Explanatory_Modeling.ipynb](Part1_Explanatory_Modeling.ipynb)

- Cleaned missing values, separating "feature absent" (no pool, no garage, no basement) from data that is truly missing
- Explored the relationship between sale price and living area, basement size, garage area, year built and lot size
- Ranked the variables most correlated with price and compared them with random forest feature importances
- Fit a linear regression on five key features to value a client's home (estimate: $164,258) and recommended improvements to raise its value

## Part II: Predictive modeling
[Part2_Predictive_Modeling.ipynb](Part2_Predictive_Modeling.ipynb)

One-hot encoded all categorical features, standardized them, and compared regularized regression models on an 80/20 train/test split.

| Model | Test R² | Test RMSE |
|---|---|---|
| Ridge (α = 100) | 0.862 | $32,586 |
| Lasso (α = 100) | 0.841 | $34,978 |
| Elastic Net (α = 50) | 0.425 | $66,393 |
| OLS (no regularization) | 0.099 | $83,126 |

Plain OLS overfits badly once all 250+ encoded features are included (train R² 0.94 vs test 0.10). Regularization fixes that, and Ridge generalizes best.

## Data
Included in this folder:
- `housing.txt`: Ames Housing dataset (also on Kaggle as *House Prices: Advanced Regression Techniques*)
- `Morty.txt`: the client home valued in Part I

Run the notebooks from this folder and they'll find the data.
