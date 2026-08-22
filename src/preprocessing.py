import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer
)

from custom_transformer import (
    log_transform,
    FeatureSelector
)


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


feature_selector = FeatureSelector()


preprocessing_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selector", feature_selector)
    ])

