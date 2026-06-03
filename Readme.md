# Machine Learning Project: Banking Marketing Campaign Prediction

## Problem Statement

A bank wants to improve the effectiveness of its telemarketing campaigns. These campaigns involve contacting customers via phone calls to promote a financial product (term deposit).

### Business Question

Can we predict whether a customer will subscribe to a term deposit based on their personal and campaign-related information?

---

# Sprint 1: Data Understanding & Preprocessing

## Step 1 : Problem Understanding & Dataset Exploration

### Dataset Overview

- Total Records: ~45,000
- Total Features: 17
- Target Variable: `target` (`yes` / `no`)
- No Missing Values
- No Duplicate Records
- Data is consistent and ready for analysis

### Feature Categories

#### Numerical Features

- age
- balance
- day
- last_call_duration
- contacts_in_campaign
- no_time_contacted_days_before
- contacted_in_before_campaing

#### Categorical Features

- job
- marital
- education
- default_credit
- housing_loan
- personal_loan
- contact_type
- month
- previous_outcome

---

## Step 2: Exploratory Data Analysis (EDA)

The objective of EDA was to understand data distributions, identify patterns, detect outliers, and study relationships between features.

### Univariate Analysis

#### Numerical Features

Performed:

- Histogram Analysis
- Distribution Analysis
- Box Plot Analysis
- Skewness Analysis

Key Findings:

- `balance` is heavily right-skewed.
- `last_call_duration` is heavily right-skewed.
- `contacted_in_before_campaing` has a very high skewness due to its count-based nature.
- Several numerical features contain outliers, but most represent genuine customer behavior.

#### Categorical Features

Performed:

- Count Plot Analysis
- Frequency Analysis

Key Findings:

- Majority of customers belong to a few dominant job categories.
- Most customers have secondary education.
- Many customers have no previous campaign outcome information.
- Target variable is imbalanced toward the `no` class.

---

### Bivariate Analysis

Performed:

- Numerical Features vs Target
- Categorical Features vs Target

Key Findings:

- Customers with longer call durations are more likely to subscribe.
- Previous successful campaign outcomes have higher subscription rates.
- Balance appears to influence customer subscription behavior.
- Loan status may affect customer responses.

---

### Multivariate Analysis

Performed:

- Correlation Analysis

Key Findings:

- No severe multicollinearity observed.
- Most numerical features show weak correlation with one another.
- Features contribute different information to the model.

---

## Step 3: Feature Engineering & Preprocessing

### Feature Transformation

The following features exhibited strong positive skewness:

- balance
- last_call_duration

Applied:

- Log Transformation on `balance`
- Log Transformation on `last_call_duration`

Generated Features:

- `balance_log`
- `duration_log`

Purpose:

- Reduce skewness
- Stabilize variance
- Improve performance for linear models

---

### Feature Selection

Selected Features:

```python
[
    'age',
    'job',
    'marital',
    'education',
    'default_credit',
    'housing_loan',
    'personal_loan',
    'contact_type',
    'contacts_in_campaign',
    'contacted_in_before_campaing',
    'previous_outcome',
    'balance_log',
    'duration_log'
]
```

### Feature Encoding

#### Ordinal Encoding

Feature:

```python
education
```

Mapping:

```python
{
    'unknown': 0,
    'primary': 1,
    'secondary': 2,
    'tertiary': 3
}
```

#### One-Hot Encoding

Applied to:

```python
job
marital
previous_outcome
contact_type
```

#### Binary Encoding

Applied to:

```python
default_credit
housing_loan
personal_loan
```

Mapping:

```python
{
    'no': 0,
    'yes': 1
}
```

---

## Key Insights Discovered

- Customers with longer call durations are more likely to subscribe.
- Previous campaign outcomes provide valuable predictive information.
- Financial indicators such as account balance may influence customer decisions.
- Most numerical features contain outliers that represent valid customer behavior rather than data errors.
- The dataset contains class imbalance, with more customers choosing not to subscribe.
- No major correlation issues were found among numerical features.

---

## Sprint 1 Outcome

Completed:

- Dataset Understanding
- Data Quality Verification
- Missing Value Analysis
- Univariate Analysis
- Bivariate Analysis
- Correlation Analysis
- Outlier Investigation
- Feature Transformation
- Feature Selection
- Feature Encoding
- Train-Test Split
- Feature Scaling 
The dataset is now prepared for machine learning model development.

---

# Sprint 2: Model Building & Evaluation

### Planned Activities

- Baseline Model Development
- Logistic Regression
- Random Forest
- XGBoost
- Model Evaluation
- Hyperparameter Tuning
- Model Comparison
- Final Model Selection

---

## Project Status

✅ Sprint 1 Completed  
🔄 Sprint 2 In Progress