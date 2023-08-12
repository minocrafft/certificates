import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss

meat_consumption_kr = 5*np.random.randn(1000) + 53.9
meat_consumption_jp = 4*np.random.randn(1000) + 32.7

meat = pd.DataFrame({"Korean": meat_consumption_kr, "Japanese": meat_consumption_jp})

print(meat.head())

plt.hist(meat_consumption_kr)
plt.xlabel("Korea")
plt.hist(meat_consumption_jp)
plt.xlabel("Japan")
plt.show()

# Z-score
meat["kr_zscore"] = ss.zscore(meat_consumption_kr)
meat["jp_zscore"] = ss.zscore(meat_consumption_jp)

print(meat.head())

# ------------------
# Scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler

std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
std = std_scaler.fit_transform(meat)
mm = mm_scaler.fit_transform(meat)

print(std)
print(mm)
