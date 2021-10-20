import pandas as pd
import matplotlib.pyplot as plt

medals_url = "http://winterolympicsmedals.com/medals.csv"
data = pd.read_csv(medals_url)
print(data.head())

def createDummies(df, var_name):
    dummy = pd.get_dummies(df[var_name], prefix=var_name)
    df = df.drop(var_name, axis = 1)
    df = pd.concat([df, dummy ], axis = 1)
    return df


#print(data.value_counts(["NOC", "Medal"]))
dt = pd.crosstab(data.Medal,data.NOC)
dt2 = pd.crosstab(data.NOC,data.Medal)

fig, axs = plt.subplots(1,2)
dt.plot(kind="pie", y="USA", use_index=True, ax=axs[0])
dt2.plot(kind="pie", y="Bronze", use_index=True, ax=axs[1])
plt.savefig("fig")
