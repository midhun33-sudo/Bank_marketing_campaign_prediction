import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Load Model Artifact

# Load Model
artifact = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "bank_marketing_final.pkl"
    )
)

# Load Scaler
scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "models",
        "scaler.pkl"
    )
)


model = artifact["model"]
threshold = artifact["threshold"]

FEATURES = [
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

def preprocess_input(data):

    df = pd.DataFrame([data])

    # Education Encoding

    education_map = {
        "unknown":0,
        "primary":1,
        "secondary":2,
        "tertiary":3
    }

    df["education"] = (
        df["education"]
        .map(education_map)
    )

    # Binary Encoding

    binary_map = {
        "yes":1,
        "no":0
    }

    df["housing_loan"] = (
        df["housing_loan"]
        .map(binary_map)
    )

    df["personal_loan"] = (
        df["personal_loan"]
        .map(binary_map)
    )

    # Log Transform

    df["balance_log"] = np.log1p(
        df["balance"] - (-8019) + 1
    )

    df["duration_log"] = np.log1p(
        df["last_call_duration"]
    )

    # One-Hot Encoding

    df["job_blue-collar"] = (
    df["job"] == "blue-collar"
    ).astype(int)

    df["job_student"] = (
        df["job"] == "student"
    ).astype(int)

    df["marital_married"] = (
        df["marital"] == "married"
    ).astype(int)

    df["marital_single"] = (
        df["marital"] == "single"
    ).astype(int)

    df["previous_outcome_other"] = (
        df["previous_outcome"] == "other"
    ).astype(int)

    df["previous_outcome_success"] = (
        df["previous_outcome"] == "success"
    ).astype(int)

    df["previous_outcome_unknown"] = (
        df["previous_outcome"] == "unknown"
    ).astype(int)

    df["contact_type_telephone"] = (
        df["contact_type"] == "telephone"
    ).astype(int)

    df["contact_type_unknown"] = (
        df["contact_type"] == "unknown"
    ).astype(int)

    df = df[FEATURES]

    num_cols = [
    "age",
    "contacts_in_campaign",
    "contacted_in_before_campaing",
    "balance_log",
    "duration_log"
    ]

    df[num_cols] = scaler.transform(
        df[num_cols]
    )

    return df


def predict_customer(data):

    processed = preprocess_input(data)

    probability = model.predict_proba(
        processed
    )[0][1]

    prediction = (
        "Will Subscribe"
        if probability >= threshold
        else "Will Not Subscribe"
    )

    return {
        "probability": round(
            float(probability),
            4
        ),

        "prediction": prediction
    }
