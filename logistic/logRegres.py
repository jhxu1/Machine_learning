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

'''
随机梯度算法
'''
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        dataMatrix[i] = array(dataMatrix[i])
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
改进的随机梯度算法
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01                  # 调整步长，缓解高频波动
            randIndex = int(random.uniform(0, len(dataIndex)))      # 随机选取样本，减少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(currLine[i])
        trainingSet.append(lineArr)
        trainingLabels.append(currLine[21])
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is : %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iteration the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    # 梯度上升算法
    # weights = gradAscent(dataMat, labelMat)
    # plotBestFit(weights.getA())      # 矩阵通过getA()将自身返回成一个n维数组对象

    # 随机梯度上升算法
    # weight = stocGradAscent0(array(dataMat), labelMat)
    # plotBestFit(weight)

    # 改进随机梯度上升算法
    # weight = stocGradAscent1(array(dataMat), labelMat)
    # plotBestFit(weight)

    # 从疝气病预测病马的死亡率
    multiTest()