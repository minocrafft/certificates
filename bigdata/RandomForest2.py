import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

df.info()
print(df.describe(include="all"))
print(df)

# for c in compare:
#     df.plot(kind="scatter", x=c, y="mpg")
#     plt.show()

df.dropna(axis=0, inplace=True)
df.drop("ocean_proximity", axis=1, inplace=True)

# 상관계수
corr = df.corr(method="pearson")
print(corr)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# regressor = DecisionTreeRegressor(max_depth=3, random_state=42)
regressor = RandomForestRegressor(max_depth=3, random_state=42)
regressor.fit(X_train, y_train)

pred = regressor.predict(X_test)
print(pred)

error = mse(y_test, pred)
print(error)
