# Predicting Customer Churn: Selecting the Optimal ML Model

## Overview

This project benchmarks five classification algorithms on the Kaggle Bank Customer Churn dataset (10,000 rows) to identify which customers are most likely to leave. Beyond model selection, the notebook investigates why accuracy is the wrong metric for imbalanced datasets and applies three recall-focused mitigation strategies — SMOTE resampling, decision-threshold tuning, and hyperparameter optimisation — to push churn detection from 49% to 66% recall.

## Business Context

On an imbalanced churn dataset (~80/20 split), a naive classifier that predicts "no churn" achieves 80% accuracy without identifying a single at-risk customer. This project benchmarks 5 algorithms and then investigates recall-focused improvements using SMOTE, threshold tuning, and hyperparameter optimisation. The target metric throughout is **recall on the churner class**: a missed churner is a lost customer.

## Data Source

Dataset: [Kaggle — Bank Customer Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) · 10,000 rows · 14 features

## Tech Stack

Python · Scikit-learn · imbalanced-learn · Pandas · Matplotlib · Seaborn · Jupyter Notebook

## Models Benchmarked

| Model | Accuracy | Churner Recall | AUC |
|---|---|---|---|
| Logistic Regression | 80.8% | 18.7% | 0.77 |
| Linear SVM | 86.1% | 39.6% | 0.83 |
| KNN | 82.4% | 34.4% | 0.75 |
| Random Forest | 86.1% | 46.0% | 0.86 |
| **Gradient Boosting** | **87.0%** | **48.9%** | **0.87** |

## Key Findings

- **Gradient Boosting is the strongest baseline** across accuracy (87.0%) and AUC (0.87), outperforming all four alternative models. Its sequential error-correction makes it better suited to the non-linear patterns in this dataset than logistic regression or distance-based methods.
- **Class imbalance suppresses recall significantly.** The default GBM catches fewer than half of churners (48.9% recall) because the training data contains ~4× more non-churners. Accuracy metrics hide this entirely.
- **SMOTE produced the largest recall gain.** Rebalancing the training set with synthetic minority examples lifted recall from 48.9% → 66.3% at a cost of ~2 accuracy points (87.0% → 84.7%). Threshold tuning at 0.35 offered a lighter-weight alternative at 60.0% recall with no retraining required.

## Mitigation Approaches

| Approach | Accuracy | Churner Recall | Notes |
|---|---|---|---|
| Default GBM (baseline) | 87.0% | 48.9% | Class imbalance unaddressed |
| SMOTE resampling | 84.7% | 66.3% | Applied to training set only |
| Threshold @ 0.35 | 85.8% | 60.0% | No retraining needed |
| Tuned GBM (GridSearchCV) | 86.6% | 48.2% | Recall-scored grid; imbalance limits gain |

## Limitations & Next Steps

- **Class imbalance:** Dataset is ~80/20. Default GBM recall on churners was 49%. Mitigation approaches tested: SMOTE, sample weighting, threshold adjustment.
- **Feature engineering:** No interaction terms or temporal features. Production deployment would benefit from recency/frequency signals.
- **Dataset:** Standard Kaggle benchmark (10k rows). Illustrative only — not representative of a specific financial institution.
- **Future work:** XGBoost comparison, Shapley value explanations, end-to-end pipeline with preprocessing.

## Project Structure

```
.
├── churn_prediction_model.ipynb   # Main analysis notebook (32 cells)
├── Churn_Modelling.csv            # Source dataset (10,000 rows)
├── gb_tuned_model.pkl             # Serialised tuned GBM (GridSearchCV best estimator)
├── images/
│   ├── class_distribution.png     # Class imbalance bar chart
│   ├── gbm_feature_importance.png # Top 10 GBM feature importances
│   ├── roc_curve_comparison.png   # ROC curves — all 5 models
│   ├── threshold_tradeoff.png     # Precision–Recall trade-off across thresholds
│   └── model_leaderboard.png      # Accuracy vs Recall grouped bar chart
└── README.md
```

## How to Run

1. Clone the repo
2. Install dependencies: `pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn`
3. Open `churn_prediction_model.ipynb` and run all cells top to bottom

---

### About the Author

**Aayush Yagol | Data Science Enthusiast & IT Professional**

Canberra-based IT professional with a Master's in Information Technology (Nov 2024) and a background in Education, transitioning into Data Science and Data Analysis. Portfolio focuses on projects that translate complex data into actionable business insights — predictive models, interactive dashboards, and data-driven applications.

- **Location:** Canberra, ACT
- **Interests:** Data Science, Photography, Music, and Fitness
