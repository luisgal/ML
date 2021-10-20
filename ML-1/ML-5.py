import numpy as np
import matplotlib.pyplot as plt

r = 1.0

pi_avg = 0
pi_value_list = []
n = 500

for j in range(1000):
    value = 0.0
    x = np.random.uniform(0,r,n).tolist()
    y = np.random.uniform(0,r,n).tolist()
    for i in range(n):
        z = np.sqrt(x[i] * x[i] + y[i] * y[i])
        if z <= r:
            value += 1.0
    pi_value = value * 4 / n
    pi_value_list.append(pi_value)
    pi_avg += pi_value

pi = pi_avg / 1000
print(pi)

plt.hist(pi_value_list, bins=10)
plt.savefig("fig4")

