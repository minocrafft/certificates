import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

df.info()
print(df.describe(include="all"))

scaler = MinMaxScaler()
df["sepal_length"] = scaler.fit_transform(df[["sepal_length"]])
df["sepal_width"] = scaler.fit_transform(df[["sepal_width"]])
df["petal_length"] = scaler.fit_transform(df[["petal_length"]])
df["petal_width"] = scaler.fit_transform(df[["petal_width"]])
print(df.describe(include="all"))

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

pred = regressor.predict(X_test)

acc = accuracy_score(y_test, pred)
print(acc)
print(confusion_matrix(y_test, pred))
print(f1_score(y_test, pred, average="weighted"))
