import numpy as np
import pandas as pd
from scipy.stats import (
    ttest_rel,  # 대응표본 t-test
    ttest_ind,  # 독립표본 t-test
    ttest_1samp,  # 단일표본 t-test
    f_oneway,  # 일원분산분석, 일원배치법
    shapiro,  # 샤피로-윌크 검정
    chi2_contingency,  # 카이제곱 검정
)


df = pd.read_csv("data/blood_pressure.csv", index_col=0)

# 1
df["diff"] = df["bp_after"] - df["bp_before"]
print(round(df["diff"].mean(), 2))

# 2
results = ttest_rel(
    df["bp_after"], df["bp_before"], alternative="less"
)  # or greater, two_sided
print(results)
print(round(results.statistic, 4))

# 3
print(round(results.pvalue, 4))

# 4
# 귀무가설 기각
