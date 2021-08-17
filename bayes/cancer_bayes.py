from posixpath import basename, splitext
from numpy.lib.shape_base import split
import pandas as pd
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

def feature_discretization(feature, slice=10):
    """特征分为10个类别"""
    assert feature.ndim == 1
    min, max = np.min(feature), np.max(feature)
    gap = (max+0.1 - min) / slice
    dis_feature = (feature - min) // gap
    return dis_feature

def algorithm(train_feature, train_label, test_feature):
    """bayes algorithm"""
    slice = 10
    for dim in range(train_feature.ndim):
        train_feature[:, dim] = feature_discretization(train_feature[:, dim], slice)
    for dim in range(test_feature.ndim):
        test_feature[:, dim] = feature_discretization(test_feature[:, dim], slice)

    label_list = [0, 1]
    assert set(train_label.tolist()) == set(label_list)

    # P(Y)
    py = {}
    for label in label_list:
        prob = sum(train_label == label) / train_label.shape[0]
        py[label] = prob
    
    # P(X|Y)
    px_y0 = {}
    px_y1 = {}
    label0_feature = train_feature[train_label == 0]
    label1_feature = train_feature[train_label == 1]
    for feature in range(slice):
        px_y0[feature] = sum(label0_feature == feature) / len(label0_feature)
        px_y1[feature] = sum(label1_feature == feature) / len(label1_feature)

    # # P(X|Y0)
    # px_in_y0 = 0
    # for feat_dim1 in range(slice):
    #     for feat_dim2 in range(slice):
    #         px_in_y0 += px_y0[feat_dim1][0] * px_y0[feat_dim2][1]
    # px_in_y1 = 0
    # for feat_dim1 in range(slice):
    #     for feat_dim2 in range(slice):
    #         px_in_y1 += px_y1[feat_dim1][0] * px_y1[feat_dim2][1]

    # pred
    # P(X|Y0)
    px_in_y0 = np.zeros((test_feature.shape[0], 1))
    for key, value in px_y0.items():
        px_in_y0[test_feature[:, 0]]
    px_y0[test_feature[:, 0]] * px_y1[test_feature[:, 1]]

    test_prob_0 = px_y0[test_feature[:, 0]] * px_y1[test_feature[:, 1]] * py[0]


    # pred_test_label = []
    # for i in range(nums):
    #     dis = ((test_feature[i, 0] - train_feature[:, 0]) ** 2 + (test_feature[i, 1] - train_feature[:, 1]) ** 2) ** 0.5
    #     k_index = np.argsort(dis)[:k]
    #     knn_label = train_label[k_index]
    #     count_set = Counter(knn_label)
    #     pred_test_label.append(count_set.most_common()[0][0])
    # return np.array(pred_test_label)

def main():
    diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test = load_data()
    pred_test_label = algorithm(diabetes_X_train, diabetes_y_train, diabetes_X_test)
    correct_num =  np.sum((diabetes_y_test - pred_test_label) == 0)
    print("Acc: {:.2f}".format(correct_num / len(diabetes_y_test)))

    # vis
    save_path = os.path.join(os.path.dirname(__file__), os.path.basename(__file__).split(".")[0])
    visualize(diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test, save_path)

if __name__ == "__main__":
    main()