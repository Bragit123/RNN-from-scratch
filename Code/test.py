import numpy as np
import pandas as pd

data = pd.read_csv("../data/clean_weather.csv", index_col=0)

x = data["tmax"].tail(3).to_numpy()

print(x)

np.random.seed(100)
Whx = np.random.normal()
Whh = np.random.normal()
Wyh = np.random.normal()
bh = 0.1*np.random.uniform()
by = 0.1*np.random.uniform()

hi = 0
for i in range(3):
    xi = x[i]
    hi = Whx*xi + Whh*hi + bh
    yi = Wyh * hi
    print(f"{i} : xi = {xi:.0f} ; hi = {hi:.0f} ; yi = {yi:.0f}")