import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

mainpath = "/app/datasets/"
filename = "customer-churn-model/Customer Churn Model.txt"

data = pd.read_csv(os.path.join(mainpath, filename))

print(len(data))

a = np.random.randn(len(data))
check = a<0.85

data_training = data[check]
data_testing = data[~check]

print(len(data_training))
print(len(data_testing))

plt.hist(a)
plt.savefig("fig6")


train, test = train_test_split(data, test_size=0.2)
print(len(train))
print(len(test))