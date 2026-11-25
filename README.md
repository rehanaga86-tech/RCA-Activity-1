# Activity 1: Multiple Linear Regression — House Price Prediction (`housing.csv`)

**Course:** Regression and Classification Algorithms (CIE 2 — Activity 1, Unit I)  
**Student:** Rehan Altaf Aga (`AD3101`)  


---

## Executive Summary

This repository presents a **Multiple Linear Regression** model trained on `housing.csv` to predict continuous residential home values (`median_house_value`) from multiple household and housing features. It works through exploratory analysis, a manual normal-equations calculation on a small sample, comparison against a single-feature baseline, multicollinearity and residual checks, and a prediction on a new example.

---

## Dataset & Problem Framing

- **Dataset File:** `housing.csv` (`20,640` rows, `10` columns)
- **Dataset Source:** California Housing Census Block-Group Dataset (`housing.csv`)
- **Target Variable (`y`):** `median_house_value` (continuous median house value in USD)
- **Selected Input Features (`X`):**
  - `median_income`
  - `housing_median_age`
  - `rooms_per_household` (`total_rooms / households`)
  - `bedrooms_per_household` (`total_bedrooms / households`)
  - `persons_per_household` (`population / households`)
- **Feature Selection & Leakage Control:** Removed `207` rows with missing `total_bedrooms` and `965` rows where `median_house_value == 500001.0` (an artificial top-coded ceiling in the dataset), trimmed extreme `0.5%` tails on per-household ratio features (`18,893` clean rows remaining), and fit `StandardScaler` strictly on the training set when checking standardized coefficients.

---

## Modeling & Validation Pipeline

- **Data Hygiene & Split:** Applied an `80:20` train-test split (`15,114` training rows and `3,779` testing rows) with `random_state=42`.
- **Baseline & Multiple Model:** Compared a single-feature baseline (`median_income`) against a 5-feature Multiple Linear Regression model on unseen test data.
- **Diagnostics & New Prediction:** Evaluated MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$, checked multicollinearity via Variance Inflation Factor (VIF), inspected residual and actual-vs-predicted plots, and predicted a new home example inside the training range.

---

## Key Results (Test Set)

On unseen test data from `housing.csv`, the 5-feature Multiple Linear Regression model beat the single-feature (`median_income`) baseline across every metric, achieving a test $R^2$ of `0.5458`, Adjusted $R^2$ of `0.5452`, and RMSE of `$65,860.79` versus `0.3862` ($R^2$), `0.3861` (Adjusted $R^2$), and `$76,560.70` (RMSE). Adding room, bedroom, occupancy, and housing-age features reduced RMSE by roughly `$10,700` while keeping multicollinearity low (`VIF <= 2.47`), confirming that the extra features genuinely improve home price predictions over income alone.

| Model | MAE ($) | MSE | RMSE ($) | $R^2$ | Adjusted $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | `78,555.18` | `9,549,795,641.00` | `97,723.06` | `0.0000` | `0.0000` |
| **Single-Feature (`median_income`)** | `57,331.20` | `5,861,540,356.91` | `76,560.70` | `0.3862` | `0.3861` |
| **Multiple Regression (5 Features)** | `48,863.69` | `4,337,643,714.85` | `65,860.79` | `0.5458` | `0.5452` |

### Fitted Equations & New Example Prediction

- **6-Row Sample Equation:** `y_hat = 151096.92 + 44736.81 * median_income - 1464.65 * housing_median_age`  
  Row 0 check: Actual `$452,600.00` | Predicted `$463,703.81` | Residual `-$11,103.81`

- **Full 5-Feature Equation:**  
  `median_house_value = 31795.52 + 58652.87 * median_income + 1457.73 * housing_median_age - 28585.21 * rooms_per_household + 127740.10 * bedrooms_per_household - 28349.15 * persons_per_household`

- **New Example Prediction:**  
  Inputs: `median_income = 4.5`, `housing_median_age = 30.0`, `rooms_per_household = 5.4`, `bedrooms_per_household = 1.08`, `persons_per_household = 2.8`  
  Predicted price: **`$243,686.78`**
---

## Decision Log, Limitations & Ethics

- **Capped Target Removal Decision:** Removed `965` rows where `median_house_value == 500001.0` so ceiling truncation would not bias least-squares slopes downward.
- **Multicollinearity & Assumption Check:** All five features had `VIF <= 2.47` (no severe multicollinearity), while residuals showed mild right skew and wider spread at higher price levels where straight-line assumptions are less exact.
- **Model Limitations:** Linear regression captures straight-line relationships and may miss non-linear effects as well as fine-grained geographic or coastal factors. A prediction is a statistical estimate—not a guaranteed price—and should not be used alone for real-world property appraisals or lending decisions.

---

## Repository Structure

- `activity_1_multiple_linear_regression_housing.ipynb` — Executed Google Colab notebook containing EDA, manual calculation steps, model training, evaluation metrics, VIF check, residual plots, and new-house prediction
- `README.md` — Project documentation, methodology, evaluation summary, and decision log
