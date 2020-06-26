import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Position_Salaries.csv")
df = pd.DataFrame(dataset)
X = df.iloc[ : , 1:2].values
Y = df.iloc[: , 2].values


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X, Y)


from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree=4)
x_poly = poly_regr.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly , Y )


#comparing the two models 
plt.scatter(X , Y , color="red")
plt.plot(X , regr.predict(X) , color="blue")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

plt.scatter(X, Y , color= "red")
plt.plot(X , lin_reg2.predict(x_poly) , color="blue")
plt.xlabel("Level")
plt.ylabel("Salary")

