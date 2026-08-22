

from ast import Import

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import classification_report

import joblib

# Loading the data
df = pd.read_csv("../data/train.csv",sep=";")

# Splitting the data
X = df.drop(columns=["y"])
y = df.y
    

# splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def log_transform(df):

    temp = df.copy()

    min_balance = -8019

    temp["balance"] = np.log1p(
        temp["balance"] - min_balance + 1
    )

    temp["duration"] = np.log1p(
        temp["duration"]
    )

    return temp



# Columns for each type of transformation   

log_cols = ["balance", "duration"]
num_cols = [
    "age",
    "campaign",
    "previous"
]

binary_cols = [
    "housing",
    "loan"
]

ordinal_cols = [
    "education"
]

nominal_cols = [
    "job",
    "marital",
    "contact",
    "poutcome"
]




# Pipelines for each type of transformation

log_pipeline = Pipeline([
    (
        "log_transform",
        FunctionTransformer(
            log_transform,
            feature_names_out="one-to-one"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])


num_pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    )
])


binary_pipeline = Pipeline([
    (
        "binary_encoder",
        OneHotEncoder(
            drop="if_binary",
            sparse_output=False,
            handle_unknown="ignore"
        )
    )
])

ordinal_pipeline = Pipeline([
    (
        "ordinal_encoder",
        OrdinalEncoder(
            categories=[
                [
                    "unknown",
                    "primary",
                    "secondary",
                    "tertiary"
                ]
            ]
        )
    )
])

nominal_pipeline = Pipeline([
    (
        "onehot",
        OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore"
        )
    )
])


# COlumn transformer to apply the pipelines to the respective columns

preprocessor = ColumnTransformer([
    (
        "num",
        num_pipeline,
        num_cols
    ),

    (
        "log_num",
        log_pipeline,
        log_cols
    ),

    (
        "binary",
        binary_pipeline,
        binary_cols
    ),

    (
        "ordinal",
        ordinal_pipeline,
        ordinal_cols
    ),

    (
        "nominal",
        nominal_pipeline,
        nominal_cols
    )
])


# Preprocessing the training data

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)   

feature_names = (preprocessor.get_feature_names_out())


X_train_processed = pd.DataFrame(X_train_processed,columns=feature_names)
X_test_processed = pd.DataFrame(X_test_processed,columns=feature_names)


# print(preprocessor.get_feature_names_out())


# Selected Features after preprocessing:
selected_features = [
    'num__age',
    'ordinal__education',
    'binary__housing_yes',
    'binary__loan_yes',
    'num__campaign',
    'num__previous',
    'log_num__balance',
    'log_num__duration',
    'nominal__job_blue-collar',
    'nominal__job_student',
    'nominal__marital_married',
    'nominal__marital_single',
    'nominal__poutcome_other',
    'nominal__poutcome_success',
    'nominal__poutcome_unknown',
    'nominal__contact_telephone',
    'nominal__contact_unknown'
]

X_train_fs = X_train_processed[selected_features]
X_test_fs = X_test_processed[selected_features]

print(X_train_fs.shape)


# Remaping the column names
rename_map = {
    'num__age':'age',
    'ordinal__education':'education',
    'binary__housing_yes':'housing_loan',
    'binary__loan_yes':'personal_loan',
    'num__campaign':'contacts_in_campaign',
    'num__previous':'contacted_in_before_campaing',
    'log_num__balance':'balance_log',
    'log_num__duration':'duration_log',
    'nominal__job_blue-collar':'job_blue-collar',
    'nominal__job_student':'job_student',
    'nominal__marital_married':'marital_married',
    'nominal__marital_single':'marital_single',
    'nominal__poutcome_other':'previous_outcome_other',
    'nominal__poutcome_success':'previous_outcome_success',
    'nominal__poutcome_unknown':'previous_outcome_unknown',
    'nominal__contact_telephone':'contact_type_telephone',
    'nominal__contact_unknown':'contact_type_unknown'
}

X_train_fs.rename(columns=rename_map, inplace=True)
X_test_fs.rename(columns=rename_map, inplace=True)

# Final order of the columns

# Feature Selection
# Rename
# Ordering

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.selected_features = [
            'num__age',
            'ordinal__education',
            'binary__housing_yes',
            'binary__loan_yes',
            'num__campaign',
            'num__previous',
            'log_num__balance',
            'log_num__duration',
            'nominal__job_blue-collar',
            'nominal__job_student',
            'nominal__marital_married',
            'nominal__marital_single',
            'nominal__contact_telephone',
            'nominal__contact_unknown',
            'nominal__poutcome_other',
            'nominal__poutcome_success',
            'nominal__poutcome_unknown'
        ]

        self.rename_map = {
            'num__age':'age',
            'ordinal__education':'education',
            'binary__housing_yes':'housing_loan',
            'binary__loan_yes':'personal_loan',
            'num__campaign':'contacts_in_campaign',
            'num__previous':'contacted_in_before_campaing',
            'log_num__balance':'balance_log',
            'log_num__duration':'duration_log',
            'nominal__job_blue-collar':'job_blue-collar',
            'nominal__job_student':'job_student',
            'nominal__marital_married':'marital_married',
            'nominal__marital_single':'marital_single',
            'nominal__poutcome_other':'previous_outcome_other',
            'nominal__poutcome_success':'previous_outcome_success',
            'nominal__poutcome_unknown':'previous_outcome_unknown',
            'nominal__contact_telephone':'contact_type_telephone',
            'nominal__contact_unknown':'contact_type_unknown'
        }

        self.final_order = [
            'age',
            'education',
            'housing_loan',
            'personal_loan',
            'contacts_in_campaign',
            'contacted_in_before_campaing',
            'balance_log',
            'duration_log',
            'job_blue-collar',
            'job_student',
            'marital_married',
            'marital_single',
            'previous_outcome_other',
            'previous_outcome_success',
            'previous_outcome_unknown',
            'contact_type_telephone',
            'contact_type_unknown'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = pd.DataFrame(X)

        X.columns = feature_names

        X = X[self.selected_features]

        X = X.rename(columns=self.rename_map)

        X = X[self.final_order]

        return X


fs = FeatureSelector()

X_train_final = fs.fit_transform(X_train_processed)
X_test_final = fs.transform(X_test_processed)

# we convert the pipeline into a full pipeline 
# by adding the feature selector , Rename and Ordering.

preprocessing_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("feature_selector", FeatureSelector())
])


X_train_final = preprocessing_pipeline.fit_transform(X_train)
X_test_final = preprocessing_pipeline.transform(X_test)


print(X_train_final.equals(X_train_fs))
print(X_test_final.equals(X_test_fs))








artifact = joblib.load(
    "../models/bank_marketing_final.pkl"
)

gb_model = artifact["model"]
threshold = artifact["threshold"]



full_pipeline = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("model", gb_model)
])

y_test = y_test.map({"no":0,"yes":1})

pred = gb_model.predict(X_test_final)
print(pred)

print(
    classification_report(
        y_test,
        pred
    )
)