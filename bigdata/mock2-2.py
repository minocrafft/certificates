import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import statsmodels.api as sm


def cut_off(pred, threshold):
    y = pred.copy()
    y[y > threshold] = 1
    y[y <= threshold] = 0
    return y.astype(int)


# -------------------------------------------------------#
#                          Exam2                         #
# -------------------------------------------------------#
seed = 42
df = pd.read_csv("parkinsons.csv")

df.info()
print(df.describe(include="all"))
print(df.head())

# 1
df.drop(["name"], axis=1, inplace=True)

for col in df.columns:
    df[col] = MinMaxScaler().fit_transform(df[[col]])

df["status"] = df["status"].astype("category")
df["const"] = 1

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=seed, stratify=y
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# model = LogisticRegression(solver="lbfgs", random_state=seed)
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
#
# score = f1_score(y_test, pred)
# print(score)

model = sm.Logit(y_train, X_train)
results = model.fit(method="bfgs", maxiter=1000)
print(results.summary())

preds = results.predict(X_test)
results = cut_off(preds, 0.5)
print(f1_score(y_test, preds))

results = cut_off(preds, 0.8)
print(f1_score(y_test, preds))
