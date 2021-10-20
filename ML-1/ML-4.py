import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5796)
fig, axs = plt.subplots(2,2)

a = 1
b = 100
n = 200000
data = np.random.uniform(a,b,n)
axs[0][0].hist(data, bins=200)

data2 = np.random.randn(n)
axs[0][1].hist(data2, bins=200)

data3 = 6 + 1*np.random.randn(n)
axs[1][0].hist(data3, bins=200)

fig.savefig("fig3")