import pandas as pd


df = pd.read_csv("data/basic1.csv")
df.info()
print(df.describe(include="all"))

# ------------------------------------------- 1
# df = df.sort_values("f5", ascending=False)
# print(df.head(10))
#
# min_value = df.iloc[:10]["f5"].min()
# df.iloc[:10, -1] = min_value
# print(df.iloc[:10])
# print(df[df["age"] >= 80]["f5"].mean())

# ------------------------------------------- 2
# indice = int(len(df) * 0.7)
# print(f"indice: {indice}")
#
# df = df.iloc[:indice]
# print(df)
#
# std1 = df["f1"].std()
# std2 = df["f1"].fillna(df["f1"].median()).std()
# print(std1, std2, std1 - std2)
