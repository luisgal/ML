import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

data = pd.read_csv("../datasets/ads/Advertising.csv")

train, test = train_test_split(data, test_size=0.2)

lm = smf.ols(formula="Sales~TV+Radio", data=train).fit()

sales_pred = lm.predict(test)

SSD = sum((test["Sales"]-sales_pred)**2)
RSE = np.sqrt(SSD/(len(test)-2-1))
error = RSE/np.mean(test["Sales"])

