import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df.info()
print(df.describe(include="all"))

# -------------------------------------------------------#
#                      Preprocessing                     #
# -------------------------------------------------------#
df["Age"].fillna(df["Age"].mean(), inplace=True)  # average
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)  # most frequent
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])
df["FamilySize"] = df["SibSp"] + df["Parch"]
df.drop(["SibSp", "Parch"], axis=1, inplace=True)
print(df)

X = df[["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# -------------------------------------------------------#
#                DecisionTreeClassifier                  #
# -------------------------------------------------------#
forest = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
forest.fit(X_train, y_train)

pred = forest.predict(X_test)

acc = accuracy_score(y_test, pred)
print(acc)
