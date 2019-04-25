# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:24:03 2019
@author: Sajal Debnath,bucse3
tutorial : course-3, Building Optimal model using Backward Elemination method.

"""
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset 
dataset = pd.read_csv('50_Startups.csv')

#dividing in x and y
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder_X = OneHotEncoder(categorical_features = [3])
X = onehotencoder_X.fit_transform(X).toarray()

#splitting dataset into train and test dataset
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#avoiding the dummy variable trap
X = X[:,1:]

#fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
ans = reg.fit(Xtrain,Ytrain)

#Predicting the train result
Y_predict = reg.predict(Xtest)

print"Prediction score is: ", ans.score(Xtest,Ytest)

#Building the optimal ,model using Backward Elimination 
import statsmodels.formula.api as sm
X = nm.append(arr = nm.ones( (50,1) ).astype(int), values=X, axis = 1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()  #Fitting using OLS
regressor_OLS.summary() #Summary for results, any feature who can be eliminated ?

#Eliminated feature 2, refitting..
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()

#Eliminated feature 1, refitting..
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()

#Eliminated feature 4, refitting..
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()

#Eliminated feature 5, refitting..
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = Y, exog= X_opt).fit()
regressor_OLS.summary()

#finished.. Optimal model found






