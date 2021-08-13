from sklearn.datasets import load_breast_cancer
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

# 取一个特征
X = X[:, 2][:, np.newaxis]