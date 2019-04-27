# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:58:37 2019

@author: Sajal
"""

#importing the libraries
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3 )

#cleanning the text
import re  # regular expression a laibrary
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range (0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


#Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Spliting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB #importing class
classifier = GaussianNB()  #create object 
classifier.fit(xTrain,yTrain)

# Predicting the Test set results
y_Predict = classifier.predict(xTest)

#making Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest,y_Predict)

