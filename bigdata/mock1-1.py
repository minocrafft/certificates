import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# -------------------------------------------------------#
#                          Exam1                         #
# -------------------------------------------------------#
df = pd.read_csv(f"data/airquality.csv")
df.info()

origin_mean = df.dropna(axis=0)["Ozone"].mean()
df["Ozone"].fillna(0, inplace=True)
new_mean = df["Ozone"].mean()
print(new_mean - origin_mean)

minmax_scaler = MinMaxScaler()
z_scaler = StandardScaler()

df["min_max"] = minmax_scaler.fit_transform(df[["Wind"]])
# z: (x - mean) / std, 아래 코드 동일
df["z"] = (df["Wind"] - df["Wind"].mean()) / df["Wind"].std()
# df["z"] = z_scaler.fit_transform(df[["Wind"]])
print(f"min_max - z: {df['min_max'].mean() - df['z'].mean()}")

print(df.groupby("Month")["Temp"].mean())
