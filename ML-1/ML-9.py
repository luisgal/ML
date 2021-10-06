import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

data = pd.read_csv("../datasets/ads/Advertising.csv")

feature_cols = ["TV", "Radio", "Newspaper"]
x = data[feature_cols]
y = data["Sales"]

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=2, step=1)
selector = selector.fit(x,y)

cols = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]

x_p = x[cols]
lm = LinearRegression()
lm.fit(x_p, y)

print(lm.score(x_p,y))
print(lm.score(x_p,y))
print(lm.score(x_p,y))
print(lm.score(x_p,y))
print(lm.score(x_p,y))