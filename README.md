📘 Behavioral Risk Prediction Model
Tech Stack:

Python • Pandas • NumPy • Scikit-learn • LightGBM • XGBoost • Seaborn • Matplotlib

🎯 Overview

This project builds a behavioral risk prediction model using a dataset of 1,884 respondents, each characterized by:

7 validated personality traits

Demographic variables (age, gender, education, ethnicity, country of residence)

Self-reported behavioral tendencies across 19 categories

The goal is to analyze these attributes and predict high-risk behavioral patterns, reframing the original dataset from a behavioral science perspective rather than a substance-related one.

This project demonstrates feature engineering, interpretability, multi-model comparison, and ensemble learning on a structured behavioral dataset.

🧠 Problem Definition

Given anonymized survey data, can we:

Model behavioral risk groups using psychometric + demographic variables?

Compare different ML algorithms and find the best model?

Understand which features influence risk classification?

Build a pipeline suitable for behavioral segmentation / risk scoring systems?

This is a classical supervised classification problem with:

Multiple target options

Highly imbalanced data

Ordinal → binary transformations

Strong reliance on feature engineering

📂 Dataset Description

The dataset includes:

🔹 Demographics

Country

Gender

Ethnicity

Age buckets

Education levels

🔹 Personality Traits

From validated psychological scales (page 8 of PDF ):

Nscore – Neuroticism

Escore – Extraversion

Oscore – Openness

Ascore – Agreeableness

Cscore – Conscientiousness

Impulsiveness

Sensation Seeking

🔹 Behavioral Indicators

19 behavioral categories originally coded as CL0–CL6 (never → last day).
We transform them into binary high-risk indicators such as:

COKE_USER

METH_USER

ECSTASY_USER

ALCOHOL_USER

(using threshold logic)

All data is anonymized and used solely for modeling and academic analysis.

🛠️ Feature Engineering

Feature engineering had major impact (as shown on PDF page 11).

Key steps:
✔ 1. Education grouping

NEW_NON_DEGREE, NEW_UNIVERSITY_DEGREE, etc.

✔ 2. Age bucketing

NEW_AGE_CAT → Young / Mid / Older

✔ 3. One-hot encoding

Country, ethnicity, gender.

✔ 4. Behavioral feature transformation

Each behavioral category → high-risk binary target.

✔ 5. Outlier handling

Tukey IQR method on numeric personality dimensions.

🔬 Models Evaluated

Using cross-validation:

Logistic Regression

KNN

SVC

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

LightGBM

Results include Accuracy, Precision, Recall, F1, ROC AUC.

🤖 Hyperparameter Optimization

Performed with GridSearchCV, with visible improvements (PDF page 13).

Example tuned parameters:

LightGBM: {'learning_rate': 0.1, 'n_estimators': 300}

XGBoost: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 200}

KNN: {'n_neighbors': 7}

🧩 Ensemble Model (VotingClassifier)

Final classifier includes:

Tuned KNN

Tuned Random Forest

Tuned LightGBM

Soft-voting significantly boosted:

Accuracy

F1

ROC-AUC

🔍 Feature Importance

LightGBM was used to interpret key drivers for each target.
PDF page 12 shows the most influential variables:

Personality scores

Country & demographic categories

Age brackets

Education groupings

Certain behavioral co-dependencies

📊 Results Summary

Ensemble models outperformed individual algorithms

Feature engineering improved performance across all metrics

Personality traits showed the strongest correlation with high-risk behavior

Models achieved strong classification performance (ROC AUC ~0.84+ depending on target)

🌍 Real-World Applications

The techniques in this project can generalize to:

Behavioral segmentation

Risk scoring (finance / insurance / fraud)

Responsible marketing

User cluster prioritization

Early-risk signal detection

Safety analytics

🚀 Future Improvements

SHAP explainability

Probability calibration

Multi-target modeling

Deploying a simple prediction API
