# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 06:22:53 2019

@author: Ryans
"""

import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
#Dividing in X and Y
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler  #class import
sc = StandardScaler()  # create object 
xTrain = sc.fit_transform(xTrain)
xTest = sc.fit_transform(xTest)

#fitting KNN to the training set
from sklearn.svm import SVC #importing class
classifier = SVC(kernel = 'rbf', random_state = 0)  #create object 
classifier.fit(xTrain,yTrain)
 
#predicting the Test set result
yPredict = classifier.predict(xTest) 

#Making the confusion matrix 
from sklearn.metrics import confusion_matrix #import function
cm = confusion_matrix(yTest, yPredict)


#visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = xTrain, yTrain
X1, X2 = nm.meshgrid(nm.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     nm.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(nm.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K Nearest Neighbour (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualizing the test set result
from matplotlib.colors import ListedColormap
X_set, y_set = xTest, yTest
X1, X2 = nm.meshgrid(nm.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     nm.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(nm.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K Nearest Neighbour (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

