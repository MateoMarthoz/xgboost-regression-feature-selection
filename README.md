# Data Science Challenge

Regression project predicting a continuous target (`outcome`) from a mixed numerical and categorical dataset (10,000 rows, 31 features). Work spans exploratory analysis, feature selection and engineering, comparison of linear and ensemble models, and hyperparameter tuning with a final XGBoost model.

## Exploratory analysis

The training data has no missing values or duplicates. The target is roughly normal; `carat` and `price` are highly skewed. Several `a` features are uniformly distributed, while `b6`–`b10` are closer to normal. Outliers appear in `carat`, `depth`, `table`, `price`, and dimensions `x`, `y`, `z`, and were addressed through transformations, trimming, and robust modelling.

PCA suggests most variance is captured in the first 30 components, so some features could be dropped without large information loss. t-SNE showed structure in `outcome` but no clear linear separation, pointing toward non-linear models.

Correlation and mutual information highlighted `depth` (strongest negative correlation with `outcome`, −0.41), with moderate signals from `b3`, `b1`, `a1`, and `a4`. Many categorical encodings and variables such as `carat` added little on their own.

Feature selection combined wrapper methods (RFE, RFA with Lasso and Random Forest) and embedded methods (Lasso, Random Forest importance). Linear and non-linear selectors disagreed in places—RFE/Lasso favoured `b3`, `b1`, `a4`, and `depth`; Random Forest emphasised `depth`, `b3`, `price`, and several `a`/`b` variables. A hybrid set was used for modelling. Automated feature engineering (Featuretools) produced additional candidates; the top 15 by correlation and XGBoost importance were retained for one branch of experiments. Selected features gave only a small test R² gain over the full set, consistent with weak pairwise relationships seen in scatter plots.

## Model selection

Five regressors were compared: Linear Regression, Ridge, Lasso, Random Forest, and XGBoost, scored with R² on train and hold-out test splits.

Linear models gave a modest baseline (test R² ≈ 0.28) but did not capture complex interactions. Random Forest and XGBoost fit the training data much more strongly yet overfit, with weaker test performance. Grid Search Cross-Validation was applied to Random Forest and XGBoost to improve generalisation (tree depth, number of estimators, learning rate, and regularisation via `colsample_bytree` and `subsample`).

XGBoost was chosen as the final algorithm for the best trade-off between fit and test performance.

## Final model and performance

On the **selected feature** set, tuned XGBoost used 200 estimators, learning rate 0.05, max depth 3, `colsample_bytree` 0.9, and `subsample` 0.8.

| Split | R² |
|-------|-----|
| Train | 0.5325 |
| Test  | 0.4614 |

A parallel experiment on heavily **processed** engineered features overfit training and generalised poorly on test, reinforcing the choice of the leaner selected feature set.

## Conclusion

XGBoost after grid-search tuning was the strongest approach, though ensemble models remained prone to overfitting. Feature selection and engineering helped only marginally, reflecting weak linear associations in the raw data. The main limitation is the gap between training and test R²; further gains would likely need different modelling strategies or stronger domain-driven features.

## Repository contents

- `notebooks/` — EDA and feature work (`01_eda`), model comparison (`02_model_selection`), tuning and predictions (`03_train_model`)
- `data/raw/` — course train and test CSVs
- `outputs/` — test-set submission (`CW1_submission_k23080165.csv`)
