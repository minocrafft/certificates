import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


X = pd.read_csv("data/X_train.csv")
y = pd.read_csv("data/y_train.csv")
test = pd.read_csv("data/X_test.csv")

X.info()
print(X.describe(include="all"))

X["환불금액"].fillna(0, inplace=True)
test["환불금액"].fillna(0, inplace=True)

for col in ["주구매상품", "주구매지점"]:
    X[col] = LabelEncoder().fit_transform(X[[col]])
    test[col] = LabelEncoder().fit_transform(test[[col]])

cust_id = test.cust_id
X.drop("cust_id", axis=1, inplace=True)
test.drop("cust_id", axis=1, inplace=True)

x_train, x_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train["gender"])

preds = model.predict(x_valid)
score = roc_auc_score(y_valid, preds)

probs = model.predict_proba(test)

output = pd.DataFrame({"cust_id": cust_id, "gender": probs[:, 1]})
output.to_csv("03000000.csv", index=False)
