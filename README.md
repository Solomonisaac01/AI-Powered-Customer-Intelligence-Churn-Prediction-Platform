# AI-Powered Customer Intelligence -- Churn Prediction Platform

A machine learning project that analyzes customer account and usage data
to predict whether a customer is likely to leave (churn). The notebook
covers the complete workflow, from exploratory data analysis and feature
engineering to model comparison, LightGBM tuning, and predictions on new
customer records.

## Project Overview

Customer churn can affect revenue, customer lifetime value, and business
growth. This project uses historical customer data to identify patterns
associated with churn and build a classification model that predicts
customer churn.

The work is implemented in a Jupyter Notebook:

`AI-Powered Customer Intelligence Churn Prediction Platform.ipynb`

> **Project scope:** This repository contains a notebook-based machine
> learning workflow and prediction example. It is not a deployed web
> application or real-time production platform.

## Problem Statement

Businesses need to identify customers who may discontinue their
services. By analyzing customer demographics, service plans, calling
behavior, charges, and customer-support interactions, a machine learning
model can estimate churn risk and support customer-retention decisions.

## Objectives

-   Understand customer data and identify patterns related to churn.
-   Clean and prepare the dataset for machine learning.
-   Explore numerical and categorical features and their relationship
    with churn.
-   Engineer and select useful predictive features.
-   Address class imbalance in the training data.
-   Compare multiple classification algorithms.
-   Tune a LightGBM model and evaluate its performance.
-   Save the trained model and demonstrate predictions on new customer
    records.

## Dataset

The notebook loads the dataset directly from the following CSV source:

[Data Science Challenge -- CSV
dataset](https://raw.githubusercontent.com/everyday-data-science/Data_Science_Projects/refs/heads/main/Sony%20Research/Data/Data_Science_Challenge.csv)

The dataset contains **3,333 customer records**, **20 input features**,
and one target column (`churn`). The notebook's initial inspection
reports no missing values and no duplicate records.

The features cover:

-   Customer information: state, account length, and area code.
-   Service plans: international plan and voice mail plan.
-   Usage: day, evening, night, and international calls and minutes.
-   Charges: day, evening, night, and international charges.
-   Support interactions: customer service calls.
-   Target: whether the customer churned.

The phone number field is removed during preprocessing because it does
not provide useful predictive information in this analysis.

## Tools & Technologies

-   **Python**
-   **Jupyter Notebook**
-   **Pandas** and **NumPy** -- data manipulation and numerical
    operations
-   **Matplotlib** and **Seaborn** -- data visualization
-   **Scikit-learn** -- preprocessing, model training, cross-validation,
    and evaluation
-   **Imbalanced-learn (ADASYN)** -- handling class imbalance
-   **XGBoost** -- gradient-boosting model comparison
-   **LightGBM** -- final tuned classification model
-   **SciPy** -- statistical analysis
-   **Joblib** -- saving and loading the trained model
-   **SHAP** -- explainability library imported in the notebook

## Project Workflow

1.  **Data loading and inspection**
    -   Load the CSV dataset.
    -   Review dimensions, data types, summary statistics, missing
        values, duplicates, outliers, and target-class distribution.
    -   Correct the area-code data type and remove the phone-number
        field.
2.  **Exploratory Data Analysis (EDA)**
    -   Analyze numerical and categorical feature distributions.
    -   Explore feature relationships and correlations.
    -   Compare customer behavior and feature values across churn
        groups.
    -   Examine potential churn indicators such as usage charges and
        customer-service calls.
3.  **Feature engineering and preprocessing**
    -   Transform selected skewed features.
    -   Bin voicemail messages and customer-service calls.
    -   Create charge-per-minute features for analysis.
    -   Split the data into training and test sets using stratification.
    -   Encode categorical variables using direct, frequency, and target
        encoding approaches.
    -   Scale numerical features with `RobustScaler`.
    -   Remove highly correlated or redundant features.
4.  **Class-imbalance handling**
    -   Apply ADASYN oversampling to the training data to address the
        minority churn class.
5.  **Model training and comparison**
    -   Compare Logistic Regression, Random Forest, Support Vector
        Classifier, Decision Tree, K-Nearest Neighbors, Gradient
        Boosting, XGBoost, LightGBM, and Gaussian Naive Bayes.
    -   Use stratified 5-fold cross-validation and classification
        metrics.
6.  **Hyperparameter tuning**
    -   Use `RandomizedSearchCV` with 5-fold cross-validation to tune
        LightGBM.
    -   Optimize using the F1 score.
7.  **Model saving and prediction**
    -   Save the tuned model with Joblib.
    -   Build a preprocessing pipeline for incoming customer records.
    -   Demonstrate churn predictions on a sample of 10 customer
        records.

## Model Performance

The notebook reports the following test-set metrics for the tuned
LightGBM model:

  Metric         Score
  ----------- --------
  Accuracy      93.85%
  Precision     87.84%
  Recall        67.01%
  F1 Score      76.02%
  ROC-AUC       90.10%

**Best parameters reported by RandomizedSearchCV:**

  Parameter           Value
  ----------------- -------
  `num_leaves`           40
  `n_estimators`        500
  `learning_rate`      0.05

**Confusion matrix counts reported in the notebook:**

                       Predicted: No Churn   Predicted: Churn
  ------------------ --------------------- ------------------
  Actual: No Churn                     561                  9
  Actual: Churn                         32                 65

These results are from the notebook's 20% held-out test set. Accuracy
alone does not describe churn detection quality; precision, recall, F1
score, and ROC-AUC are also included to provide a broader evaluation.

## Installation & Setup

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL with your GitHub repository URL.

### 2. Install dependencies

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm scipy joblib shap jupyter
```

### 3. Launch Jupyter Notebook

``` bash
jupyter notebook
```

Open `AI-Powered Customer Intelligence Churn Prediction Platform.ipynb`
and run the cells in order.

The notebook downloads the dataset from the CSV URL above, so an
internet connection is required when loading the data.

## Repository Structure

``` text
AI-Powered-Customer-Churn-Prediction/
│
├── AI-Powered Customer Intelligence Churn Prediction Platform.ipynb
├── README.md
└── best_lgb_model.pkl        # Optional: include if you upload the saved model
```

The notebook saves the trained model as `best_lgb_model.pkl` when the
model-saving cell is run. Add this file to the repository only if you
want to share the trained model; it is not included automatically by
this README.

## Business Impact

A churn prediction model can help a business:

-   Identify customers who may be at risk of leaving.
-   Prioritize retention outreach and customer-support efforts.
-   Understand service usage and customer-support patterns associated
    with churn.
-   Support data-driven customer-retention strategies.

Predictions should be treated as decision-support signals, not as proof
that a customer will leave.

## Skills Demonstrated

-   Data cleaning and preprocessing
-   Exploratory data analysis
-   Statistical analysis and feature correlation
-   Feature engineering and feature selection
-   Categorical encoding and numerical scaling
-   Handling imbalanced datasets with ADASYN
-   Supervised machine learning and classification
-   Cross-validation and hyperparameter tuning
-   Model evaluation and interpretation of classification metrics
-   Building a preprocessing pipeline
-   Saving and loading machine learning models
-   Generating predictions for new data

## Project Outcome

Developed and evaluated a customer-churn classification workflow using
multiple machine learning algorithms. The tuned LightGBM model achieved
**93.85% accuracy**, **76.02% F1 score**, and **90.10% ROC-AUC** on the
notebook's held-out test set. A prediction function was also implemented
to demonstrate churn predictions for new customer records.

## Author

**Solomon Isaac**

-   GitHub: [Solomonisaac01](https://github.com/Solomonisaac01)

------------------------------------------------------------------------

If you find this project useful, consider giving the repository a ⭐.
