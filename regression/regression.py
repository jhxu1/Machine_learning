from numpy import  *

'''
功能：读取文件数据
'''
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

'''
功能：计算最佳拟合直线
输入：属性数组（m*n） 类别数组（m*1）
输出：ws（n*1）
'''
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat                             # 计算X^T*X
    if linalg.det(xTx) == 0:
        print("This matrix is singular, cannot do inverse")         # 行列式为0，矩阵不可逆
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

'''
功能：lwlr：局部加权线性回归
输入：testPoint(xi) 属性数组（m*n） 类别数组（m*1）
输出：testPoint的预测值(1*n)
'''
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))            # w = (X^TWX)^(-1)X^TWy
    return testPoint * ws

'''
功能：加权线性回归
输入：测试数组（m*n) 属性数组（m*n） 类别数组（m*1）
输出：预测数组(m*n)
'''
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

'''
功能：岭回归求解
'''
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

'''
功能：数据标准化处理及岭回归测试
'''
def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)              # 求每个特征的均值
    xVar = var(xMat, 0)                 # 求每个特征的无偏估计
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat

if __name__ == '__main__':
    # # 线性回归绘制
    # xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr, yArr)
    # import matplotlib.pyplot as plt
    # xMat = mat(xArr); yMat = mat(yArr)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])          # flatten()把数据降到一唯 矩阵.A = 矩阵.getA()
    # xCopy = xMat.copy()
    # xCopy.sort(0)                                                               # 按行排序
    # yHat = xCopy * ws
    # ax.plot(xCopy[:, 1], yHat)
    # plt.show()
    # # 计算相关系数
    # yHat = xMat * ws
    # cor = corrcoef(yHat.T, yMat)
    # print(cor)

    # # 局部加权线性回归
    # xArr, yArr = loadDataSet('ex0.txt')
    # yHat = lwlrTest(xArr, xArr, yArr, 0.03)
    # print(yHat)
    # xMat = mat(xArr)
    # strInd = xMat[:, 1].argsort(0)      # 依据第二个属性排序的索引列表
    # xSort = xMat[strInd][:, 0, :]       # 获取X的排序
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:, 1], yHat[strInd])
    # ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    # plt.show()

    # 预测鲍鱼的年龄
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)                   # 根据图形选择最优的lambda
    plt.show()
