from math import *
from numpy import *
import numpy as np

'''
读取数据
'''
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

'''
sigmoid函数
'''
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''
功能：梯度上升算法
输入：dataMatIn（2位Numpy数组） classLabels（类别标签，1x100行向量）
输出：
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()         # 转化为Numpy矩阵数据类型并转置
    m, n = shape(dataMatrix)
    alpha = 0.001   # 步长
    maxCycles = 500     # 迭代次数
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2,ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]     # 设定0=w0x0+w1x1+w2x2,解出x1和x2的关系式
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    # 梯度上升算法
    # weights = gradAscent(dataMat, labelMat)
    # plotBestFit(weights.getA())      # 矩阵通过getA()将自身返回成一个n维数组对象

    # 随机梯度上升算法
    weight = stocGradAscent0(dataMat, labelMat)
    plotBestFit(weight)