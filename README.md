# 🩺 Diabetes Risk Prediction Using Machine Learning

This project aims to predict diabetes risk using machine learning techniques based on health indicator data from the **Behavioral Risk Factor Surveillance System (BRFSS 2015)** dataset.

---

## 📁 Dataset

- **Source**: [BRFSS 2015](https://www.cdc.gov/brfss/index.html)
- **File**: `diabetes_binary_health_indicators_BRFSS2015.csv`
- **Target Column**: `Diabetes_binary`
    - `1` = Has Diabetes
    - `0` = No Diabetes
- **Features**: 21 health indicators including BMI, smoking status, physical activity, cholesterol levels, etc.

---

## 🔍 Project Objectives

- Load and explore the BRFSS dataset
- Perform data cleaning and preprocessing
- Visualize class distribution
- Train and evaluate multiple ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting Classifier
  - Naive Bayes
- Compare performance metrics
- Conclude with insights and recommendations

---

## 📊 Exploratory Data Analysis

- Checked for missing values and class imbalance
- Visualized diabetes vs non-diabetes distribution
- Explored feature relationships using heatmaps and pair plots

---

## 🧠 Machine Learning Models

| Model                 | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Gradient Boosting    | 0.840    | 0.428     | 0.472  | 0.449    |
| Logistic Regression  | 0.731    | 0.310     | 0.775  | 0.443    |
| Naive Bayes          | 0.724    | 0.295     | 0.719  | 0.418    |
| Random Forest        | 0.850    | 0.434     | 0.286  | 0.345    |
| Decision Tree        | 0.796    | 0.292     | 0.334  | 0.312    |

✅ **Best model** based on F1 Score: **Gradient Boosting Classifier**

---

## 📌 Key Findings

- The dataset is slightly imbalanced (more non-diabetic cases).
- Gradient Boosting provides the most balanced performance.
- Logistic Regression shows high recall, useful for sensitive predictions.

---

## 📎 Requirements

- Python 3.x
- Jupyter Notebook
- pandas, matplotlib, seaborn
- scikit-learn

Install using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
