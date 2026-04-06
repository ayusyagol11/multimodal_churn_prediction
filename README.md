# Predicting Customer Churn: Selecting the Optimal ML Model

## Overview

A multi-model machine learning benchmark on the Kaggle Bank Customer Churn dataset (10,000 rows) that evaluates six classification algorithms across nine configurations. The project goes beyond accuracy to investigate recall-focused improvements using SMOTE, sample weighting, threshold adjustment, and hyperparameter tuning.

## Business Context

On an imbalanced churn dataset (~80/20 split), a naive classifier that predicts no churn achieves 80% accuracy without identifying a single at-risk customer. The real metric is recall on the churn class — how many at-risk customers does the model actually catch?

## Tech Stack

Python · Scikit-learn · imbalanced-learn · Pandas · Matplotlib · Seaborn · Jupyter Notebook

## Data Source

[Kaggle — Bank Customer Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) · 10,000 rows · 14 features

## Model Benchmark Results

| Model | Accuracy | Recall (Churn) | Precision (Churn) | F1 (Churn) | AUC |
|---|---|---|---|---|---|
| GBM (Weighted) | 80.0% | **75.9%** | 50.6% | 60.7% | 0.869 |
| GBM (SMOTE) | 84.7% | 66.3% | 61.5% | 63.8% | 0.867 |
| GBM | 87.0% | 48.9% | 79.3% | 60.5% | 0.871 |
| Hist GBM | 86.0% | 48.4% | 73.8% | 58.5% | 0.857 |
| GBM (Tuned) | 86.6% | 48.2% | 77.5% | 59.4% | 0.867 |
| Random Forest | 86.1% | 46.0% | 76.3% | 57.4% | 0.855 |
| SVM (RBF) | 86.1% | 39.6% | 83.4% | 53.7% | 0.827 |
| KNN | 82.4% | 34.4% | 62.2% | 44.3% | 0.753 |
| Logistic Regression | 80.8% | 18.7% | 58.9% | 28.4% | 0.775 |

*Sorted by Recall (Churn) descending — the primary business metric.*

## Key Findings

- GBM achieved the highest accuracy (87.0%) among base models, but only 48.9% recall on churners at the default threshold
- Sample weighting produced the largest recall improvement: 48.9% → 75.9%, though at the cost of ~7 accuracy points; SMOTE (66.3% recall) offers a better precision-recall balance for capacity-constrained retention teams
- Threshold adjustment to 0.35 improved recall without retraining (60.0% recall)
- GridSearchCV optimised for recall produced marginal gains, suggesting class imbalance is the primary bottleneck — not hyperparameters

## Visualisations

| Chart | Description |
|---|---|
| `images/customer_churn_class_distribution.png` | Class imbalance bar chart (79.6% / 20.4%) |
| `images/gbm_feature_importance_top_10_predictors.png` | Top 10 GBM feature importances |
| `images/roc_curve_comparison_all_6_base_models.png` | ROC curves — all 6 base models |
| `images/precision_vs_recall_tradeoff_across_thresholds.png` | Score vs threshold (recall / precision / F1) |
| `images/precision_recall_tradeoff_decision_thresholds_default_gbm.png` | Precision–Recall tradeoff scatter |
| `images/recall_improvement_gbm_mitigation_approaches.png` | Accuracy vs Recall across 4 GBM variants |
| `images/model_leaderboard_accuracy_vs_recall_catch_rate.png` | Accuracy vs Recall grouped bar — all 6 models |

## Limitations & Next Steps

- **Class imbalance:** Dataset is ~80/20. SMOTE + threshold tuning pushes churner recall to ~66% — further improvement would require richer features
- **Feature engineering:** No interaction terms or temporal features. Production would benefit from recency/frequency/monetary signals
- **Dataset:** Standard Kaggle benchmark (10k rows). Illustrative only
- **Future work:** XGBoost/LightGBM comparison, SHAP value explanations, end-to-end sklearn Pipeline with preprocessing

## Project Structure

```
.
├── churn_prediction_model.ipynb   # Main analysis notebook (36 cells)
├── Churn_Modelling.csv            # Source dataset (10,000 rows)
├── gb_tuned_model.pkl             # Serialised tuned GBM (GridSearchCV best estimator)
├── images/
│   ├── customer_churn_class_distribution.png
│   ├── gbm_feature_importance_top_10_predictors.png
│   ├── model_leaderboard_accuracy_vs_recall_catch_rate.png
│   ├── precision_recall_tradeoff_decision_thresholds_default_gbm.png
│   ├── precision_vs_recall_tradeoff_across_thresholds.png
│   ├── recall_improvement_gbm_mitigation_approaches.png
│   └── roc_curve_comparison_all_6_base_models.png
├── .gitignore
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
