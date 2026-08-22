"""
Training script for the Bank Marketing term-deposit subscription model.

Fixes applied vs. the original version:
1. `y_train` was never mapped to {0, 1} (only `y_test` was) — this silently breaks any
   imbalance-handling step (SMOTE, class_weight) and most sklearn classifiers' assumptions
   about label encoding. Both splits are now mapped consistently.
2. The original script trained `full_pipeline` (a fresh GradientBoostingClassifier) but then
   evaluated and reported metrics for `final_pipeline`, which wrapped a DIFFERENT, previously
   saved model loaded from disk — so the printed classification_report never reflected the
   model that was actually just trained.
3. The original script depended on `../models/bank_marketing_final.pkl` already existing
   (to load `gb_final_model`) before it could even build `final_pipeline` — an impossible
   chicken-and-egg dependency on a fresh checkout / first run.
4. The newly trained model and its fitted preprocessing pipeline were never saved to disk
   (the `joblib.dump` call was commented out), so `predict.py` had nothing fresh to load.

This script now trains one model, evaluates exactly that model, and persists both the fitted
preprocessing pipeline and the final model artifact that `predict.py` depends on.
"""
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preprocessing import preprocessing_pipeline

DATA_PATH = "../data/train.csv"
MODEL_OUTPUT_PATH = "../models/bank_marketing_final.pkl"
PIPELINE_OUTPUT_PATH = "../models/preprocessing_pipeline.pkl"
DECISION_THRESHOLD = 0.60  # selected via threshold optimization in the notebook (Sprint 3)

LABEL_MAP = {"yes": 1, "no": 0}


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH, sep=";")

    # 2. Segregate features / target
    X = df.drop(columns=["y"])
    y = df["y"].map(LABEL_MAP)

    # 3. Split — both y_train and y_test are mapped to {0, 1} consistently
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Fit preprocessing on training data only, transform both splits
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    # 5. Train the model (Gradient Boosting was selected as the final model — see notebook)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_processed, y_train)

    # 6. Evaluate the model we JUST trained — not a different, previously saved one
    test_probs = model.predict_proba(X_test_processed)[:, 1]
    test_preds = (test_probs >= DECISION_THRESHOLD).astype(int)

    print(f"Evaluation at decision threshold = {DECISION_THRESHOLD}")
    print(classification_report(y_test, test_preds))

    # 7. Persist the fitted preprocessing pipeline (predict.py loads this directly,
    #    instead of re-fitting or re-implementing encoding logic by hand)
    joblib.dump(preprocessing_pipeline, PIPELINE_OUTPUT_PATH)

    # 8. Persist the final model artifact
    final_artifact = {
        "model_name": "Gradient Boosting",
        "model": model,
        "threshold": DECISION_THRESHOLD,
        "features": list(X_train_processed.columns),
        "metrics": {
            "accuracy": classification_report(y_test, test_preds, output_dict=True)["accuracy"],
        },
    }
    joblib.dump(final_artifact, MODEL_OUTPUT_PATH)

    print(f"\nSaved model artifact to: {MODEL_OUTPUT_PATH}")
    print(f"Saved fitted preprocessing pipeline to: {PIPELINE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
