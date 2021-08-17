from posixpath import basename, splitext
from numpy.lib.shape_base import split
from sklearn import datasets
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import random
import os
import pdb

def visualize(diabetes_X_train, diabetes_y_train, diabetes_X_test, pred_y_label, save_path):
    # color
    label_choice = set(diabetes_y_train.tolist())
    color_map = {l: (random.random(), random.random(), random.random()) for l in label_choice}
    diabetes_y_colors = np.zeros((diabetes_y_train.shape[0], 3))
    pred_y_colors = np.zeros((pred_y_label.shape[0], 3))
    for key, value in color_map.items():
        diabetes_y_colors[diabetes_y_train == key] = value
        pred_y_colors[pred_y_label == key] = value

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Train")
    axs[0].scatter(diabetes_X_train[:, 0], diabetes_X_train[:, 1], c=diabetes_y_colors)
    axs[1].set_title("Pred")
    axs[1].scatter(diabetes_X_test[:, 0], diabetes_X_test[:, 1], c=pred_y_colors)
    fig.savefig(save_path)
  

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

    # vis
    save_path = os.path.join(os.path.dirname(__file__), os.path.basename(__file__).split(".")[0])
    visualize(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test, save_path)

if __name__ == "__main__":
    main()