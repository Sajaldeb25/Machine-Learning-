# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 08:21:08 2019

@author: Sajal Debnath
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
#Dividing in X and Y
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,[2]].values

#Fitting the DTR to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)

#predicting a new result
y_predict = regressor.predict(6.5)
y_predict = regressor.predict(8)

#visualizing the DTR result
plt.scatter(X,Y,color = 'red')
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,Y,color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff(Decisionn Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
