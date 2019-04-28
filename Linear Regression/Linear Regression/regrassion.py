# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:56:11 2019
@author: Sajal Debnath
"""
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

#importing the data set
dataset = pd.read_csv('Beton.csv')

x =  dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values 

#splitting the data set into training set and test set
#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 0)

#fitting simple linear regeression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(xTrain, yTrain)

#predicting the test set result
y_predict = reg.predict(xTest)
 
## The line / model
plt.scatter(xTrain, yTrain, color='red')
plt.plot(xTrain,reg.predict(xTrain),color='blue')
plt.title('X vs y graph')
plt.xlabel("Value of x")
plt.ylabel("Value of y")
plt.show()

print "Prediction score is :", model.score(xTest, yTest)
