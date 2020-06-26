import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("50_Startups.csv")
df = pd.DataFrame(dataset)
X = df.iloc[: , :-1 ].values
Y = df.iloc[: , 4].values

#encoding categorical values
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
Y = LabelEncoder().fit_transform(Y)

#splitting the training and test result
X_train , Y_train , X_test , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=0)

#calling regressor method 
regr = LinearRegression()
regr.fit(X_train , Y_train)

y_pred = regr.predict(X_test)


plt.scatter(X_train , Y_train , color="red")
plt.plot(X_train , regr.predict(X_test) , color ="blue")
plt.xlabel("R&D,Administration , marketin spend")
plt.ylabel("Profit")
plt.show()

plt.scatter(X_test , Y_test , color="red")
plt.plot(X_train , regr.predict(X_test) , color ="blue")
plt.xlabel("R&D,Administration , marketin spend")
plt.ylabel("Profit")
plt.show()






