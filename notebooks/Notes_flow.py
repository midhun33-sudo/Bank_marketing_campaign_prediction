import numpy as np
import matplotlib.pyplot as plt 
# import seaborn as sns
import pandas as pd
# import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,log_loss,roc_auc_score
import joblib


path = "../data/train.csv"
df = pd.read_csv(path,sep=";")

# Standardizing the Column Names
# changing the names of columns to understandable format
df.rename(columns={"y":"target"}, inplace=True)
df.rename(columns={"default":"default_credit"}, inplace=True)
df.rename(columns={"housing":"housing_loan"}, inplace=True)
df.rename(columns={"loan":"personal_loan"}, inplace=True)
df.rename(columns={"contact":"contact_type"}, inplace=True)
df.rename(columns={"poutcome":"previous_outcome"}, inplace=True)
df.rename(columns={"pdays":"no_time_contacted_days_before"}, inplace=True)
df.rename(columns={"duration":"last_call_duration"}, inplace=True)
df.rename(columns={"campaign":"contacts_in_campaign"}, inplace=True)
df.rename(columns={"previous":"contacted_in_before_campaing"}, inplace=True)


# names has changed 
num_col = df.select_dtypes(include=["int64","float64"]).columns
cat_col = df.select_dtypes(include=["object"]).columns
# print(num_col)
# print(cat_col)


# trasfromation of the data
df["balance_log"] = np.log1p(df["balance"] - df["balance"].min() + 1)
df["duration_log"] = np.log1p(df["last_call_duration"])




x = df.drop(columns=["balance","day","month","last_call_duration","no_time_contacted_days_before","target"])


# Encoding

# Label Encoding
maps = {"no": 0,"yes": 1}

x["default_credit"] = x["default_credit"].map(maps)
x["housing_loan"] = x["housing_loan"].map(maps)
x["personal_loan"] = x["personal_loan"].map(maps)

# ordinal encoding 
education_map = {'unknown':0,'primary':1,'secondary':2,'tertiary':3}
x["education"] = x["education"].map(education_map)

# one hot encoding
one_hot_cols = ["job","marital","previous_outcome","contact_type"]
one_hot_encoded = pd.get_dummies(x[one_hot_cols],dtype=int,drop_first=True)


# Making X and Y Data Values
x = pd.concat([x.drop(columns=one_hot_cols),one_hot_encoded],axis=1)
y = df["target"].map(maps)


# splitting
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

# Feature Scaling 
scaler = StandardScaler()

num_cols = ["age",
            "contacts_in_campaign",
            "contacted_in_before_campaing",
            "balance_log",
            "duration_log"]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])



print("X - Training length :",len(X_train))
print("X - Testing length :",len(X_test))
print("y - Training length :",len(y_train))
print("y - Testing length :",len(y_test))








low_features = [
    'job_housemaid',
    'job_management',
    'job_unknown',
    'default_credit',
    'job_technician',
    'job_self-employed',
    'job_entrepreneur',
    'job_retired',
    'job_unemployed',
    'job_services'
]

X_train_fs = X_train.drop(columns=low_features)
X_test_fs = X_test.drop(columns=low_features)



artifact = joblib.load(
    "../models/bank_marketing_final.pkl"
)

model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]




pred = model.predict(X_test_fs)

print(
"Accuracy", accuracy_score(y_test,pred),
"Precision", precision_score(y_test,pred),
"Recall", recall_score(y_test,pred),
"F1", f1_score(y_test,pred))