import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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






# Final order of the columns

# Feature Selection
# Rename
# Ordering

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names = [
            'num__age',
            'num__campaign',
            'num__previous',
            'log_num__balance',
            'log_num__duration',
            'binary__housing_yes',
            'binary__loan_yes',
            'ordinal__education',
            'nominal__job_blue-collar',
            'nominal__job_entrepreneur',
            'nominal__job_housemaid',
            'nominal__job_management',
            'nominal__job_retired',
            'nominal__job_self-employed',
            'nominal__job_services',
            'nominal__job_student',
            'nominal__job_technician',
            'nominal__job_unemployed',
            'nominal__job_unknown',
            'nominal__marital_married',
            'nominal__marital_single',
            'nominal__contact_telephone',
            'nominal__contact_unknown',
            'nominal__poutcome_other',
            'nominal__poutcome_success',
            'nominal__poutcome_unknown'
        ]

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
        # scikit-learn's check_is_fitted() (used internally by Pipeline.transform())
        # looks for any fitted attribute ending in "_" on the estimator. Custom
        # transformers with no learned state still need to expose one of these,
        # otherwise calling pipeline.transform() after fit_transform() raises
        # NotFittedError even though the transformer has nothing to learn.
        self.n_features_in_ = (
            X.shape[1] if hasattr(X, "shape") else len(self.feature_names)
        )
        return self

    def transform(self, X):

        X = pd.DataFrame(X)

        X.columns = self.feature_names

        X = X[self.selected_features]

        X = X.rename(columns=self.rename_map)

        X = X[self.final_order]

        return X

