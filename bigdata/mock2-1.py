import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------#
#                          Exam1                         #
# -------------------------------------------------------#
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.info()
print(df.describe(include="all"))
print(df)

# 1
print(df["Attrition"].value_counts())
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
print(df["Attrition"].value_counts())

# 2
categories = df.select_dtypes("object", "category").columns.values
print("nunique: ", df[categories].nunique())
df.drop(["Over18"], axis=1, inplace=True)
df.info()

# 3
categories = df.select_dtypes("int64").columns
new_df = df[categories]
corr_df = new_df.corr(method="pearson")
print(corr_df[np.abs(corr_df) >= 0.9].info())
condition = ["JobLevel", "MonthlyIncome"]
new_df.drop("JobLevel", axis=1, inplace=True)
print(new_df)
