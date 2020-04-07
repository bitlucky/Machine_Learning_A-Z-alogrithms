#importing libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")
df = pd.DataFrame(dataset)
X = df.iloc[: , :-1].values
Y = df.iloc[: , 1].values

#train test split 
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=0)

# regressor object to predict result 
regressor = LinearRegression()
regressor.fit(X_train , Y_train )

y_pred = regressor.predict(X_test)

#visualizing the result 

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
