# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 18:28:31 2019

@author: Sajal Debnath
"""

import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset 
dataset = pd.read_csv('Position_Salaries.csv')

#dividing in x and y
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#fitting Linear regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#fitting Polynomial regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X) 

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

#visualising the Linear regression results
plt.scatter(X,Y,color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title('Checking the truthness by Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#visualising the Polynomial Regression result 
X_grid = nm.arange(min(X), max(X)+1, .001)
#reshape for reshaping the data into a len(X_grid)*1 array, to make a column out of the X_grid values 
X_grid = X_grid.reshape(len(X_grid),1) 

plt.scatter(X,Y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Checking the truthness by Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with linear Regression
lin_reg.predict(6.5)
#predicting a new result with polynomial regression 
lin_reg_2.predict(poly_reg.fit_transform(6.5))
lin_reg_2.predict(poly_reg.fit_transform(1))



 
