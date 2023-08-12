import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale


df = pd.read_csv("data/mtcars.csv")

df.info()
print(df.describe())
print(df.head())

# df["qsec"] = MinMaxScaler().fit_transform(df[["qsec"]])
df["qsec"] = minmax_scale(df["qsec"])

target = df[df["qsec"] > 0.5]
print(target)
print(len(target))
