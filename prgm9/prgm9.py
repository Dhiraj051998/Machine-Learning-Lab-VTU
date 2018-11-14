"""
Write a program to implement K-Nearest Neighbour algorithm to classify the iris
data set. Print both correct and wrong predictions. Java/Python ML library classes can
be used for this problem.
"""

from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

dataset = datasets.load_iris()
X = pd.DataFrame(dataset.data)
y = pd.DataFrame(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train.values.ravel())
y_prediction = knn.predict(X_test)

print('K-Nearest Neighbour Accuracy : ', metrics.accuracy_score(y_test, y_prediction))
print("Confusion Matrix : \n", metrics.confusion_matrix(y_test, y_prediction))
