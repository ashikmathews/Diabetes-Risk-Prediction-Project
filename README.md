# 🩺 Diabetes Risk Prediction - Machine Learning Project

## 📌 Overview

This project focuses on **predicting diabetes risk** using health indicators from the **BRFSS 2015 dataset**. The objective is to build, evaluate, and compare multiple machine learning models to determine which provides the best performance in identifying individuals at risk of diabetes.

---

## 🎯 Problem Statement

**Goal**: To develop a machine learning model that can accurately predict whether a person has diabetes (binary classification: 0 = No Diabetes, 1 = Has Diabetes) based on their health attributes.

---

## 📂 Dataset Used

- **Source**: CDC - Behavioral Risk Factor Surveillance System (BRFSS) 2015
- **File Name**: `diabetes_binary_health_indicators_BRFSS2015.csv`
- **Total Records**: ~253,000
- **Features**: 21
- **Target Variable**: `Diabetes_binary`

---

## 🧪 Steps Performed

### 1. Data Loading and Initial Exploration
- Loaded the dataset using `pandas`
- Checked for missing values, data types, and class imbalance

### 2. Exploratory Data Analysis (EDA)
- Visualized diabetes distribution (how many have vs don't have diabetes)
- Analyzed relationships between features like BMI, age, smoking, blood pressure, etc.
- Found clear trends in how health behaviors correlate with diabetes

### 3. Data Preprocessing
- Removed duplicate records
- Performed feature scaling with `StandardScaler`
- Split data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`)

### 4. Model Building and Training
Trained the following models:
- ✅ Logistic Regression
- 🌲 Random Forest
- 🌿 Decision Tree
- 🚀 Gradient Boosting Classifier
- 📊 Naive Bayes
- (Optional) KNN & SVM (if time and compute allowed)

### 5. Model Evaluation
Evaluated models using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score

Visualized performance comparison using bar charts.

### 6. Results Summary

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Gradient Boosting    | 0.84     | 0.43      | 0.47   | **0.45** |
| Logistic Regression  | 0.73     | 0.31      | 0.77   | 0.44     |
| Naive Bayes          | 0.72     | 0.29      | 0.71   | 0.42     |
| Random Forest        | 0.85     | 0.43      | 0.29   | 0.34     |
| Decision Tree        | 0.79     | 0.29      | 0.33   | 0.31     |

✔️ **Best Model Based on F1 Score**: **Gradient Boosting Classifier**

---

## 📈 Key Findings

- The dataset is imbalanced: majority class is people **without diabetes**
- Health factors like **BMI**, **High Blood Pressure**, **Cholesterol**, and **Age Category** are highly correlated with diabetes
- Among all models tested, **Gradient Boosting Classifier** had the best balance of Precision, Recall, and F1-Score

---

## 📚 What You’ll Learn from This Project

- How to clean and analyze real-world health datasets
- Binary classification using multiple ML models
- How to interpret classification metrics
- How to choose the best model for a health prediction problem

---

## 🧠 Tools & Libraries Used

- Python 3.8+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (sklearn)

---

## ✅ Conclusion

This project demonstrates the effectiveness of machine learning in healthcare analytics, particularly for early prediction of diabetes risk using survey-based health indicators. With models like **Gradient Boosting**, we can achieve reliable predictions that may support public health planning and preventive interventions.

---

## 📌 How to Run

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
