"""
Inference script for the Bank Marketing term-deposit subscription model.

IMPORTANT CHANGE FROM THE ORIGINAL VERSION:
This script now reuses `preprocessing_pipeline` from `preprocessing.py` — the exact same
pipeline used during training (see `train.py` and the rebuilt notebook) — instead of
re-implementing encoding logic by hand. This guarantees inference and training can never
silently drift apart.

Practical effect: `predict_customer()` now expects the RAW UCI Bank Marketing schema
(the same column names that come straight out of `train.csv`), not pre-renamed/pre-engineered
fields like `housing_loan` or `balance_log`. See RAW_INPUT_EXAMPLE below for the expected shape.

It also no longer loads a separate `scaler.pkl` — scaling is handled inside
`preprocessing_pipeline`, so that file is not needed anymore.
"""
import os

import joblib
import pandas as pd
import mlflow


MLRUNS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "notebooks", "mlruns")
)
mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocessing_pipeline  # noqa: E402  (import after BASE_DIR setup)

# ---------------------------------------------------------------------------
# Load the trained model artifact
# ---------------------------------------------------------------------------
artifact = joblib.load(os.path.join(BASE_DIR, "models", "bank_marketing_final.pkl"))

model = artifact["model"]
threshold = artifact["threshold"]
FEATURES = artifact["features"]  # exact column order the model was trained on

# The preprocessing_pipeline itself must already be fitted on training data before this
# script can transform new data. In production, persist and load the FITTED pipeline
# (e.g. joblib.dump(preprocessing_pipeline, "models/preprocessing_pipeline.pkl") right
# after training) rather than re-fitting it here. See the note in `train.py`.
fitted_pipeline = joblib.load(os.path.join(BASE_DIR, "models", "preprocessing_pipeline.pkl"))

# Raw UCI schema expected as input to preprocess_input() / predict_customer()
RAW_INPUT_EXAMPLE = {
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 220,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def preprocess_input(data: dict) -> pd.DataFrame:
    """Transforms a single raw customer record into the model's expected feature format.

    `data` must use the RAW UCI column names (age, job, marital, ..., poutcome) —
    the same schema as a row of `train.csv` minus the target column `y`.
    """
    df = pd.DataFrame([data])
    processed = fitted_pipeline.transform(df)

    # Defensive check: make sure the pipeline output matches what the model expects,
    # in both columns present AND order, before handing it to model.predict_proba().
    return processed[FEATURES]


def predict_customer(data: dict) -> dict:
    """Returns the subscription probability and a thresholded prediction for one customer."""
    processed = preprocess_input(data)

    probability = model.predict_proba(processed)[0][1]
    prediction = "Will Subscribe" if probability >= threshold else "Will Not Subscribe"

    return {
        "probability": round(float(probability), 4),
        "prediction": prediction,
    }


if __name__ == "__main__":
    result = predict_customer(RAW_INPUT_EXAMPLE)
    print(result)
