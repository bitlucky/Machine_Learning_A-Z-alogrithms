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

