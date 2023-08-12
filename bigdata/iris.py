import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

df.info()
print(df.describe())

df["species"].replace({"setosa": 0, "versicolor": 1, "virginica": 2}, inplace=True)
print(df.head())

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

pred = tree.predict(X_test)
print(pred)

acc = accuracy_score(y_test, pred)
print(acc)
print(confusion_matrix(y_test, pred))
print(f1_score(y_test, pred, average="weighted"))
