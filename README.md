# insurance-pricing-model-xgboost
Скоринг-модель, позволяющая оценить индивидуальный риск каждого водителя (вероятность ДТП) и на его основе рассчитать справедливую стоимость полиса.

_**Язык программирования Python!**_

This project builds an end-to-end machine learning pipeline to predict insurance risk and optimize pricing.

The model estimates:
- Claim probability (frequency)
- Claim severity (amount)
- Expected loss
- Optimal premium to achieve target loss ratio (70%)

The solution includes data cleaning, feature engineering, model training, calibration, and business-level pricing optimization.

## Data Availability
The dataset is not publicly available due to confidentiality reasons.


##  Pipeline Overview

### 1. Data Cleaning
- Handling invalid values (experience, car year, bonus-malus)
- Missing value imputation

### 2. Feature Engineering
- Time-based features (year, month, quarter)
- Risk indicators (bonus-malus transformations)
- Driver experience groups
- Interaction features

### 3. Aggregations
- Driver-level statistics (claim rate, policy count)
- Policy-level features

### 4. SCORE Feature Compression
- Aggregated into grouped averages
- Reduced dimensionality

### 5. Feature Selection
- IV (Information Value)
- VIF (multicollinearity check)
- XGBoost feature importance (95% threshold)

### 6. Modeling
- Frequency model (XGBoost classifier)
- Severity model (XGBoost regressor)

### 7. Calibration
- Probability calibration to match real claim rate
- Severity bias correction

### 8. Pricing Optimization
- Expected loss calculation
- Premium adjustment to achieve target loss ratio (70%)

### 9. Evaluation
- ROC-AUC
- Gini coefficient
- Residual analysis
- Stability tests

## Model Performance Results

### Frequency Model (Claim Probability)

- CV AUC (validation): **0.8212**
- Gini coefficient: **0.6424**
- Train AUC: **0.8915**

### Severity Model (Claim Amount)

- R² (log-scale): **0.7595**
- Log-bias correction: **0.0038**
- Average predicted severity:
  - Train: **462,272 KZT**
