from numpy import *
from math import *

def loadSimpData():
    dataMat = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

'''
功能：通过阈值比较对数据进行分类
输入：dataMatrix（属性矩阵）dimen（第几列）threshVal（阈值）threshIneq（不等式类型）
输出：m*1的向量
'''
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

'''
功能：找到最佳单层决策树
输入：dataArr（属性矩阵）classLabels（类别向量）D（样本权重）
输出：bestStump（最佳决策树信息）minError（最小权重误差）besstClassEst（最优样本预测类别）
'''
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps               # 求步长
        for j in range(-1, int(numSteps)+1):                    # 将阈值设置为取值范围之外
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)    # 设置阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)    # 通过阈值比较对数据类别进行预测
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, \
                #       inequal, weightedError))
                if weightedError < minError:        # 如果错误率小于最小错误率
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i                        # 第几个特征
                    bestStump['thresh'] = threshVal             # 阈值
                    bestStump['ineq'] = inequal                 # 大于还是小于
    return bestStump, minError, bestClassEst


'''
功能：基于单层决策树的AdaBoost训练过程
输入：dataArr（属性矩阵）classLabels（类别向量）numIt（最大迭代次数）
输出：AdaBoost分类器（每次迭代的信息）
'''
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)                     # 权重向量初始化为相等值
    aggClassEst = mat(zeros((m, 1)))
    print('---------------------------开始训练AdaBoost分类器----------------------------')
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D: ", D.T)
        alpha = float(0.5 * log((1-error)/max(error, 1e-16)))               # max(error, 1e-16)的作用是防止error为0的情况
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:  ", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)         # 计算e^-alpha
        for i in range(m):
            D[i] = multiply(D[i], exp(expon[i]))                            # 更改每个样本的权重
        D = D/sum(D)
        aggClassEst += alpha * classEst                                     # 通过aggClassEst了解总的分类
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = float(aggErrors.sum()) / m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    print('---------------------------训练AdaBoost分类器完成----------------------------')
    return weakClassArr

'''
功能：AdaBoost预测
输入：datToClass（属性）classifierArr（AdaBoost分类器）
输出：预测类别
'''
def adaClassify(datToClass, classifierArr):
    print('---------------------------开始预测数据',datToClass,'----------------------------')
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    for i in range(len(aggClassEst)):
        aggClassEst[i] = sign(aggClassEst[i])
    print('---------------------------预测结果为\n', aggClassEst, '\n----------------------------')
    return aggClassEst


'''
功能：读取文件数据
输入：fileName（文件名）
输出：属性矩阵 类别向量
'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))           # 获取特征数量(numFeat-1)
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[-1]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat




if __name__ == '__main__':
    # 数据点可视化
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # dataMat, classLabels = array(loadSimpData())
    # ax.scatter(dataMat[:, 0], dataMat[:, 1])
    # plt.show()

    # 单层决策树生成函数
    # dataMat, classLabels = loadSimpData()
    # D = mat(ones((5, 1))/5)
    # bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    # print("最优单层决策树:\n", bestStump, "\n最小权重误差:\n", minError, "\n最优样本预测类别:\n", array(bestClassEst))

    # 构造AdaBoost分类器及测试
    dataMat, classLabels = loadSimpData()
    weakClassArr = adaBoostTrainDS(dataMat, classLabels)
    adaClassify([0,0], weakClassArr)

    # 使用AdaBoost预测马的疝气病
    # dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    # classifierArray = adaBoostTrainDS(dataArr, labelArr, 10)
    # testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # prediction10 = adaClassify(testArr, classifierArray)
    # errArr = mat(ones((67, 1)))
    # err = errArr[prediction10 != mat(testLabelArr).T].sum()
    # print(err)
