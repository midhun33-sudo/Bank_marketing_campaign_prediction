# Machine Learning Project: Banking Marketing Campaign Prediction

## Problem Statement

A bank wants to improve the effectiveness of its telemarketing campaigns. These campaigns involve contacting customers via phone calls to promote a financial product (term deposit).

### Business Question

Can we predict whether a customer will subscribe to a term deposit based on their personal and campaign-related information?

---

## Project Objectives

* Analyze customer and campaign-related data.
* Identify factors influencing customer subscription decisions.
* Build baseline and advanced classification models.
* Compare model performance using multiple evaluation metrics.
* Select the best-performing model for deployment.

---

## Dataset Information

The dataset contains customer demographic information, financial attributes, and details about previous marketing interactions.

### Features Include:

* Age
* Job
* Marital Status
* Education
* Balance
* Housing Loan
* Personal Loan
* Contact Type
* Campaign Information
* Previous Campaign Outcome

### Target Variable

* y

  * Yes → Customer subscribed
  * No → Customer did not subscribe

---

## Project Workflow

### Sprint 1 : Data Preparation & Exploration

#### Data Understanding

* Dataset inspection
* Data type validation
* Missing value analysis

#### Exploratory Data Analysis

* Univariate Analysis
* Bivariate Analysis
* Multivariate Analysis

#### Data Cleaning

* Handling missing values
* Duplicate checking
* Outlier analysis

#### Feature Engineering

* Encoding categorical variables
* Feature transformation
* Data preparation for modeling

---

### Sprint 2 : Model Building & Evaluation

#### Baseline Model

* Logistic Regression

#### Multiple Model Training

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Other classification models

#### Model Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

#### Model Comparison

* Performance comparison across models
* ROC-AUC analysis
* Best model selection

---

## Results

The models were evaluated using multiple classification metrics to ensure robust performance.

Key focus areas included:

* Reducing false negatives
* Improving recall
* Maximizing ROC-AUC performance
* Ensuring model generalization

The final selected model demonstrated the best balance between predictive performance and business requirements.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost

### Development Environment

* Jupyter Notebook
* Git
* GitHub

---

## Project Structure

```text
Banking-Marketing-Classification/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── Banking_marketing_ML_project.ipynb
│
├── Readme.md
│
└── study_note
```

---

## Key Learnings

* End-to-end machine learning workflow
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering techniques
* Classification model development
* Model evaluation and comparison
* Business-driven machine learning decision making

---

## Future Improvements

* Hyperparameter tuning
* Model deployment using Flask/FastAPI
* Interactive dashboard development
* Automated retraining pipeline
* Cloud deployment

---

## Author

K.Midhun Kumar

Machine Learning | Data Analytics | Software Development

Focused on building practical data-driven solutions and continuously improving machine learning and analytical skills.
