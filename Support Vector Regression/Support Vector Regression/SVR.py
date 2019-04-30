# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 11:47:54 2019

@author: SAJAL Debnath
"""
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset 
dataset = pd.read_csv('Position_Salaries.csv')

#dividing in x and y
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,[2]].values

#feature scaling 
from sklearn.preprocessing import StandardScaler   
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(X,Y)

#predicting a new result 
y_predict = regressor.predict(6.5)

#visualising the Linear regression results
plt.scatter(X,Y,color='red')
plt.plot(X, regressor.predict(X),color='blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()






 
