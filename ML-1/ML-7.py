import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

data = pd.read_csv("../datasets/ads/Advertising.csv")
lm = smf.ols(formula="Sales~TV", data=data).fit()

print(lm.summary())

sales_pred = lm.predict(pd.DataFrame(data["TV"]))
print(sales_pred)

data.plot(kind="scatter", x="TV", y="Sales")
plt.plot(pd.DataFrame(data["TV"]), sales_pred, c="red", linewidth=2)

plt.savefig("fig7")

data["sales_pred"] = pd.DataFrame(sales_pred)
SSD = sum((data["sales_pred"]-data["Sales"])**2)

RSE = np.sqrt(SSD/(len(data)-2))

error = RSE/np.mean(data["Sales"])

print("SSD: ", SSD)
print("RSE: ", RSE)
print("error: ", error)

print(data.corr())
