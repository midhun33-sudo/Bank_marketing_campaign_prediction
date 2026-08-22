import pandas as pd
from preprocessing import preprocessing_pipeline
from sklearn.model_selection import train_test_split


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

import joblib



# 1. Loading the data
df = pd.read_csv("../data/train.csv",sep=";")

# 2. Segrigating the data
X = df.drop(columns=["y"])
y = df.y
    

# 3. Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_test = y_test.map({"yes":1,"no":0})
# y_train = y_train.map({"yes":1,"no":0})


# 4. Creating the model 
gb_model = GradientBoostingClassifier(random_state=42)


artifact = joblib.load(
    "../models/bank_marketing_final.pkl"
)
gb_final_model = artifact["model"]

# 5. Creating the full pipeline

final_pipeline = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("model", gb_final_model)
])


full_pipeline = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("model", gb_model)
])



# 6. Training the model

# full_pipeline.fit(X_train, y_train)
full_pipeline.fit(X_train,y_train)

# 7. validating the model

# pred = full_pipeline.predict(X_test)

pred = final_pipeline.predict(X_test)
print(classification_report(y_test, pred))



# 8. Saving the model

# joblib.dump(
#     full_pipeline,
#     "../models/bank_marketing_pipeline.pkl"
# )

# Testing with the Preprocessing pipeline
# X_train_final = preprocessing_pipeline.fit_transform(X_train)

# print(X_train_final.shape)
# print(X_train_final.columns)
