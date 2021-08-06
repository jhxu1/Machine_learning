import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt 

X, y = load_breast_cancer(return_X_y=True)

# 取一个特征
X = X[:, 2][:, np.newaxis]

# train,test
X_train, X_test = X[:-30], X[-30:]
y_train, y_test = y[:-30], y[-30:]

clf = RidgeClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_test)
print("Train Set score: {}".format(clf.score(X_train, y_train)))
print("Test Set score: {}".format(clf.score(X_test, y_test)))

