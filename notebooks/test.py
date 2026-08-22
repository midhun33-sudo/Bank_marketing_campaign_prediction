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



import joblib
pipeline = joblib.load(
    "bank_marketing_pipeline.pkl"
)

pipeline.predict(X_train)