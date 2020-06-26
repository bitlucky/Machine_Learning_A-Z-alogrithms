import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
df = pd.DataFrame(dataset)
X = df.iloc[: , 1:2 ].values
Y = df.iloc[: , 2].values

#Standard Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
Y = sc_y.fit_transform(Y)

from sklearn.svm import SVR
svr = SVR()
svr.fit(X, Y)


