from numpy import *
import operator


def createDateSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


'''
实现KNN分类算法
输入：inX(样本属性）dataSet（训练数据属性）labels（训练数据标签）k（从前k个最近的样本选择）
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

if __name__ == '__main__':
    group, labels = createDateSet()
    result = classify0([0,0],group,labels,3)
    print(result)