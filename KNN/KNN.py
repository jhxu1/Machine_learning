from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDateSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
实现KNN分类算法
输入：inX(样本属性）dataSet（训练数据属性）labels（训练数据标签）k（从前k个最近的样本选择）
输出：输入样本的预测标签
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                  # 样本数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet     # 计算与每一个样本中每一个属性的差，diffMat为一个dataSetSize*m的向量(m为属性个数)
    sqDiffMat = diffMat ** 2                            # 平方
    sqDistances = sqDiffMat.sum(axis=1)             # 行求和，得到每一个样本的距离
    distance = sqDistances ** 0.5                   # 开根
    sortedDistIndicies = distance.argsort()         # 排序，并将索引保存
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 获取第i个样本的标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1      # 获取该标签的已有数量并加一，若不存在先置0，再加一
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)     # 从大到小排序
    return sortedClassCount[0][0]


'''
功能：读取datingTestSet2文件中的数据，并存为矩阵形式
输入：filename（文件名）
输出：属性矩阵 标签向量
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)        # 获取文件行数
    returnMat = zeros((numberOfLines, 3))   # 创建返回的矩阵
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()                 # 移除换行符
        listFromLine = line.split('\t')     # 按照制表符划分
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


'''
功能：归一化属性矩阵
输入：dataSet（属性矩阵）
输出：归一化矩阵 最小值和最大值差值向量 最小值向量
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0)        # 获取每列的最小值
    maxVals = dataSet.max(0)        # 获取每列的最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]            # 计算矩阵行数
    normDataSet = dataSet - tile(minVals, [m, 1])
    normDataSet = normDataSet/tile(ranges, [m, 1])
    return normDataSet, ranges, minVals

'''
功能：测试分类器分类效果
输入：无
输出：错误率
'''
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normalMat, ranges, minVals = autoNorm(datingDataMat)
    m = datingDataMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normalMat[i, :], normalMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    errorRate = errorCount/numTestVecs*1.0
    print("the total error rate is: %d%%" %int(errorRate*100))
    return errorRate


def classifyPerson():
    resultList = ['不喜欢的人', '魅力一般的人', '极具魅力的人']
    ffMiles = float(input("你每年的飞行里程数\n"))
    percentTats = float(input("你每天玩游戏时间占一天的百分比是多少\n"))
    iceCream = float(input("每一年吃掉多少升冰淇淋\n"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0(inArr, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult])



if __name__ == '__main__':
    datingClassTest()
    classifyPerson()
    # datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
    # plt.show()
