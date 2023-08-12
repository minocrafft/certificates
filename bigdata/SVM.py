import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# -------------------------------------------------------#
#                           SVM                          #
# -------------------------------------------------------#
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df)
df.info()
print(df.describe(include="all"))

df["Age"].fillna(df["Age"].mean(), inplace=True)  # average
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # most frequent
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])
df["FamilySize"] = df["SibSp"] + df["Parch"]
df.drop(["SibSp", "Parch"], axis=1, inplace=True)
print(df)

onehot_sex = pd.get_dummies(df["Sex"])
df = pd.concat([df, onehot_sex], axis=1)

onehot_embarked = pd.get_dummies(df["Embarked"])
df = pd.concat([df, onehot_embarked], axis=1)
df.info()

X = df[["Pclass", "Age", "Fare", "FamilySize", "female", "male", "C", "Q", "S"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)

pred = svm.predict(X_test)
acc = accuracy_score(y_test, pred)
print(acc)

report = classification_report(y_test, pred)
print(report)

help(SVC)
