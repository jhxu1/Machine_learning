from numpy import *

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
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, \
                      inequal, weightedError))
                if weightedError < minError:        # 如果错误率小于最小错误率
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


if __name__ == '__main__':
    # 数据点可视化
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # dataMat, classLabels = array(loadSimpData())
    # ax.scatter(dataMat[:, 0], dataMat[:, 1])
    # plt.show()

    dataMat, classLabels = loadSimpData()
    D = mat(ones((5, 1))/5)
    bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    print("最优单层决策树:\n", bestStump, "\n最小权重误差:\n", minError, "\n最优样本预测类别:\n", array(bestClassEst))