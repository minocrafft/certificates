import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer


# -------------------------------------------------------#
#                          Exam2                         #
# -------------------------------------------------------#
df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
df.info()
print(df)
print(df.describe(include="all"))

df.drop(["ID", "ZIP Code"], axis=1, inplace=True)

X = df.drop(["Personal Loan"], axis=1)
y = df["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = Normalizer().fit_transform(X_train)
X_test = Normalizer().transform(X_test)

train_acc = []
test_acc = []

for n in range(1, 25):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, y_train)

    train_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))

plt.plot(range(1, 25), train_acc, label="Train Acc")
plt.plot(range(1, 25), test_acc, label="Test Acc")
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()

print(test_acc)
max_k = np.argmax(test_acc) + 1
acc = test_acc[max_k - 1]
print(max_k, acc)
