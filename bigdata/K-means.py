import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

df.info()
print(df.describe(include="all"))

df["species"] = LabelEncoder().fit_transform(df["species"])
print(df["species"].unique())

ddf = df
sns.pairplot(ddf, hue="species")
plt.show()

cluster = KMeans(
    n_clusters=3, n_init=10, max_iter=500, random_state=42, algorithm="auto"
)
cluster.fit(df)
cluster_centers = cluster.cluster_centers_
cluster_prediction = cluster.predict(df)
print(pd.DataFrame(cluster_centers))
print(cluster_prediction)

ddf["cluster"] = cluster_prediction
print(ddf)

inertias = []
for k in range(1, 10):
    model = KMeans(
        n_clusters=k, n_init=10, max_iter=500, random_state=42, algorithm="lloyd"
    )
    model.fit(df)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 6))

plt.plot(range(1, 10), inertias, "-o")
plt.xlabel("number of clusters, k")
plt.ylabel("inertia")
plt.show()
