import pandas as pd
from sklearn import datasets
import numpy as np
from collections import Counter
import pdb


def load_data():
    diabetes_X, diabetes_y = datasets.load_breast_cancer(return_X_y=True)
    # Use only one feature
    diabetes_X = diabetes_X[:, :2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]
    return diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test

def knn_algorithm(train_feature, train_label, test_feature, k=10):
    nums = len(test_feature)
    pred_test_label = []
    for i in range(nums):
        dis = ((test_feature[i, 0] - train_feature[:, 0]) ** 2 + (test_feature[i, 1] - train_feature[:, 1]) ** 2) ** 0.5
        k_index = np.argsort(dis)[:k]
        knn_label = train_label[k_index]
        count_set = Counter(knn_label)
        pred_test_label.append(count_set.most_common()[0][0])
    return np.array(pred_test_label)

def main():
    diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = load_data()
    pred_test_label = knn_algorithm(diabetes_X_train, diabetes_y_train, diabetes_X_test)
    correct_num =  np.sum((diabetes_y_test - pred_test_label) == 0)
    print("Acc: {:.2f}".format(correct_num / len(diabetes_y_test)))

if __name__ == "__main__":
    main()