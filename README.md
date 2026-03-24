# 🏦 Bank Customer Churn Prediction: An Ensemble Learning Approach and Multimodal Comparison

### 📊 Project Overview
This project focuses on predicting customer churn for a financial institution using a dataset of 10,000 customers. The goal was to build a predictive "early warning system" that identifies at-risk customers, allowing the business to intervene before they walk out the door.

### 🔗 Data Source
The data for this project was sourced from the **shrutimechlearn Churn Modelling dataset** on Kaggle.
* **Dataset Link:** [Kaggle - Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

### ⚙️ Technical Highlights
* **Preprocessing:** Handled categorical data using **Label Encoding** and **One-Hot Encoding** (drop_first=True) to prevent the Dummy Variable Trap.
* **Feature Engineering:** Identified **Age** as the most significant predictor of churn through Random Forest Feature Importance.
* **Ensemble Learning:** Compared five models, ranging from linear baselines to sequential boosting techniques.

### 📈 Model Leaderboard & Evaluation

| Rank | Model | Accuracy | Recall (Class 1) | Precision (Class 1) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 | **GBM (Gradient Boosting)** | **86.75%** | **49%** | 75% | **Winner.** Highest catch rate for churners. |
| 🥈 | **Random Forest** | 86.75% | 47% | **76%** | **Strong Runner-Up.** Most precise. |
| 🥉 | **KNN** | 83.00% | 37% | 61% | Decent local cluster detection. |
| 4th | **LogReg** | 81.00% | 20% | 55% | Weak. Fails to capture non-linear trends. |
| 5th | **Linear SVM** | 80.35% | 0% | 0% | Fail. Predicted zero churn. |

**Key Finding:** For churn prediction, we prioritize **Recall** over Accuracy. Identifying nearly half of all churners (GBM) provides significantly more business value than a model that is technically accurate but fails to identify at-risk customers.

---

### 👨‍💻 About Me
**Aayush Yagol | Data Science Enthusiast & IT Professional**

I am a Canberra-based IT professional with a **Master’s in Information Technology (Nov 2024)** and a background in Education. I am currently leveraging my technical foundation in IT and my analytical mindset to transition into **Data Science and Data Analysis**.

My portfolio focuses on projects that translate complex data into actionable business insights, such as:
* **Claims Liability & Insurance Churn Predictive Models**
* **Interactive Wildfire Dashboards**
* **Personal Finance Tracker Web Applications**

I am passionate about using machine learning to solve real-world problems, with a particular interest in predictive modeling and data visualization.

* **Location:** Canberra, ACT 🇦🇺
* **Interests:** Data Science, Photography, Music, and Fitness.

---

### 🚀 How to Run
1. Clone the repo.
2. Ensure you have `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn` installed.
3. Open `churn_prediction_model.ipynb` and run all cells.

