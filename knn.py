from numpy import double
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas
import numpy as np
from sklearn.datasets import load_iris

def Knn():
    X = []
    y = []
    f = open("model/keypoint_classifier/keypoint.csv", "r")
    for i in f.readlines():
        X.append(list(map(float, i[10:-1].split(","))))
        y.append(int(i[0]))
    #print(X[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train = OneHotEncoder().fit_transform(X_train)
    X_test = OneHotEncoder().fit_transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn = knn.fit(X,y)
    #print(knn.score(X_test,y_test))
    #print("KNN Predicts: " + str(knn.predict([data,data])[0]))
    """
    iris = load_iris()

    X = iris.data
    print(X)

    y = iris.target
    print(y)

    X_train, X_test, y_train, y_test = train_test_split(
                 X, y, test_size = 0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=7)
    print(knn)

    knn.fit(X_train, y_train)

    print(knn.score(X_test, y_test))
    """
    return knn