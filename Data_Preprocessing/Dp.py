import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("Data.csv")
X= dataset.iloc[:,:-1].values
df_x = pd.DataFrame(X)
Y=dataset.iloc[:,3].values
df_y= pd.DataFrame(Y)

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN", strategy="mean" , axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.fit_transform(X[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y= LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#splitting the training test and test set
from sklearn.model_selection import train_test_split
X_train, Y_train , X_test, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
Y_train=sc_X.fit_transform(X_test)

