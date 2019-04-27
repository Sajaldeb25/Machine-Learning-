# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:07:26 2019

@author: Sajal
"""

import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Shoppers.csv')
#Dividing in X and Y
X = dataset.iloc[:, [3,4]].values
# Y = dataset.iloc[:, 4].values
'''
#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler  #class import
sc = StandardScaler()  # create object 
xTrain = sc.fit_transform(xTrain)
xTest = sc.fit_transform(xTest)
'''

# using the elbow method to find the optional number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# fitting KMeans to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++',  random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'CLuter 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('CLusters of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1–100)')
plt.legend()
plt.show()




#fitting KNN to the training set
from sklearn.ensemble import RandomForestClassifier  #importing class
classifier = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0 )  #create object 
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

