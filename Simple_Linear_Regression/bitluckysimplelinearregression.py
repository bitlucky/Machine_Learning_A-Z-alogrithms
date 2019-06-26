import numpy as np
import pandas as pd
import matplotlib.pyplot as plt# -*- coding: utf-8 -*-

# import the datasets
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[: , :-1].values
Y= dataset.iloc[: , 1].values

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test =train_test_split(X,Y , test_size= 1/3, random_state = 0)

#fitting simple linear regression to the training set 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , Y_train)

#prediction the test set result 

y_pred = regressor.predict(X_test)

#visualizing the training set result

plt.scatter(X_train , Y_train , color= "red")
plt.plot(X_train , regressor.predict(X_train),color= "blue")
plt.title("experience vs salary")
plt.xlabel("years of experience(training set)")
plt.ylabel("salary")
plt.show()


plt.scatter(X_test, Y_test , color = "red")
plt.plot(X_train , regressor.predict(X_train),color = "blue")
plt.title("experience vs salary(test set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

