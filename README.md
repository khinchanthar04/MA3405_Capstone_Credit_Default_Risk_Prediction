# Optimal Statistical Learning Models for Credit Default Risk Prediction

**Khin Chan Thar**

---

## 📌 Overview

This project benchmarks five machine learning classification models to identify the most effective approach for predicting credit card default. Using a dataset of 30,000 credit card clients in Taiwan, the analysis covers the full pipeline from data wrangling, feature engineering, unsupervised exploration, supervised model training, and multi-threshold performance evaluation, to arrive at a deployment recommendation for credit risk systems.

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| Source | UCI Machine Learning Repository (Yeh, 2009) |
| Records | 30,000 credit card clients (Taiwan) |
| Features | 23 independent variables |
| Target | `default.payment.next.month` (Binary: 1 = Default, 0 = No Default) |
| Class Split | ~78% No Default / ~22% Default |

**Feature Categories:**
- **Demographics:** Gender, Education, Marital Status, Age
- **Credit & Finance:** Credit Limit (`LIMIT_BAL`), 6 months of bill statements (`BILL_AMT1–6`), 6 months of payment amounts (`PAY_AMT1–6`)
- **Payment History:** 6 months of repayment status (`PAY_0–PAY_6`) — the most critical predictors

---

## 🛠️ Methodology

### Data Preparation
1. **Feature Engineering** — Simplified 11-level `PAY_*` variables into three categories: `Normal`, `Delay_1_2M`, `Severe_Delay`
2. **Standardisation** — Z-score normalisation applied to all 12 continuous predictors using training set statistics
3. **Train/Test Split** — 70% training (21,000 records) / 30% test (9,000 records)

### ⚙️ Unsupervised Learning
- **PCA** — Applied to 12 standardised numeric predictors to reduce dimensionality; first 3 PCs capture ~61% of variance
- **K-Means Clustering** — 3 clusters fitted on PC1–PC3, visualised on the PC1 vs PC2 plane to identify natural customer segments

### 🏷️ Supervised Classification Models

| Model | Approach |
|---|---|
| **Logistic Regression** | Baseline linear model; GLM with binomial family |
| **LASSO** | Penalised logistic regression (α = 1) with cross-validated λ for feature selection and multicollinearity control |
| **Classification Tree (CART)** | Gini impurity criterion; interpretable rule-based model |
| **Random Forest** | Ensemble of 300 decision trees via bagging and random subspace feature selection |
| **SVM (RBF Kernel)** | Non-linear decision boundary in high-dimensional space; trained on 30% subset due to computational cost |

### 🎯 Evaluation Metrics
All models evaluated at probability thresholds of **0.3, 0.5, and 0.7**:
- **Accuracy** — Overall correct prediction rate
- **Sensitivity** — True Positive Rate (correctly identified defaulters) — primary metric for credit risk
- **Specificity** — True Negative Rate (correctly identified non-defaulters)
- **AUC** — Discrimination ability across all thresholds

---

## 📝 Results

### Performance at Multiple Thresholds

| Model | Threshold | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Logistic | 0.3 | 0.777 | 0.504 | 0.855 |
| Logistic | 0.5 | 0.804 | 0.279 | 0.954 |
| LASSO | 0.3 | 0.777 | 0.505 | 0.855 |
| LASSO | 0.5 | 0.804 | 0.277 | 0.955 |
| Classification Tree | 0.5 | 0.777 | 0.000 | 1.000 |
| **Random Forest** | **0.3** | **0.775** | **0.555** | **0.838** |
| Random Forest | 0.5 | 0.806 | 0.349 | 0.937 |
| SVM Radial | 0.3 | 0.803 | 0.396 | 0.920 |
| SVM Radial | 0.5 | 0.799 | 0.273 | 0.950 |

### AUC Comparison

| Model | AUC |
|---|---|
| **Random Forest** | **0.785** |
| Logistic Regression | 0.750 |
| LASSO | 0.749 |
| SVM Radial | 0.730 |
| Classification Tree | 0.629 |

### 🔎 Key Findings
- **Random Forest** achieved the highest AUC (0.785) and the highest Sensitivity (0.555 at threshold 0.3), making it the top-performing model overall.
- **PAY_0** and **PAY_2** (recent payment status) were by far the most important predictors — outweighing all bill amounts and demographic variables in the variable importance plot.
- All models exhibited low Sensitivity at the standard 0.5 threshold due to the **78/22 class imbalance**, with the models biased toward predicting the majority (No Default) class.
- The **Classification Tree** produced zero sensitivity at thresholds ≥ 0.5, confirming its unsuitability for this imbalanced task.

---

## 🎯 Conclusion & Recommendations

The **Random Forest** model is recommended for deployment due to its superior AUC and sensitivity. Implementation guidance:

1. **Deploy Random Forest** as the primary classification model.
2. **Lower the classification threshold to 0.3** to improve detection of high-risk customers, accepting a moderate reduction in specificity.
3. **Address class imbalance** in future work using SMOTE (Synthetic Minority Over-sampling Technique) or cost-sensitive learning to further improve defaulter detection rates.
4. **Monitor PAY_0 and PAY_2** as primary early warning indicators in production.

---

## Tech Stack

- **Language:** R
- **Key Packages:** `glmnet`, `randomForest`, `e1071`, `tree`, `pROC`, `caret`, `factoextra`, `ggplot2`, `dplyr`

---

## Project Structure

```
├── 14684190_Khin_Chan_Thar_Report.docx   # Full written report
├── MA3405_Result.docx                     # R code output and model results
└── README.md                              # This file
```

---

## Data Source

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

---

## Author

**Khin Chan Thar**  
Credit Default Risk Prediction — Statistical Learning Capstone
