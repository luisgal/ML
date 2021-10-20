import pandas as pd
import matplotlib.pyplot as plt

medals_url = "http://winterolympicsmedals.com/medals.csv"

data = pd.read_csv(medals_url)
print(data.columns.values)

data1 = data[data["NOC"]=="USA"][["Medal"]]
print(data1)

plt.hist(data1)
plt.savefig("fig2")


