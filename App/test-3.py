import pandas as pd
w =16
h =14
xs = list(range(1,w+1))
ys = list(range(1, h+1))
coordinates = [(str(x),str(y)) for x in xs for y in ys]
index = pd.MultiIndex.from_tuples(coordinates)
qtable = pd.DataFrame(index = index, columns= range(1,5), dtype= int)
qtable = qtable.fillna(0)
print(qtable)
print(qtable.loc[("1", "3"), 1])