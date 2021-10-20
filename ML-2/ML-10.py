import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("../datasets/ecom-expense/Ecom Expense.csv")

dumy_gender = pd.get_dummies(data["Gender"], prefix="Gender")
dumy_city_tier = pd.get_dummies(data["City Tier"], prefix="City")

df = data.join(dumy_gender).join(dumy_city_tier)

feature_colums = ["Monthly Income", "Transaction Time",
                  "Gender_Female", "Gender_Male",
                  "City_Tier 1", "City_Tier 2", "City_Tier 3"]

X = df[feature_colums]
Y = df["Total Spend"]

lm = LinearRegression()
lm.fit(X,Y)

co = df.corr()