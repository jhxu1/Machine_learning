from numpy import *


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList,classVec

'''
功能：获取数据集中的不重复词表
输入：dataSet（数据集）
输出：词汇表
'''
def createVocabList(dataSet):
    vocabSet = set([])      # 创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

'''
功能：获取一个list，表示词汇表中的数据是否在输入数据集中出现
输入：vacabList（词汇表） inputSet（输入数据集）
输出：词汇表中的数据是否在输入数据集中出现，0（未出现）1（出现）
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word %s is not in my Vocabylary!" % word)
    return returnVec


'''
功能：词袋模型实现上述功能
'''
def bagOfwords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word %s is not in my Vocabylary!" % word)
    return returnVec


'''
功能：朴素贝叶斯训练器
输入：trainMatrix（属性矩阵，类别向量）
输出：每种属性的条件概率即类别概率
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)     # 获取样本数
    numWords = len(trainMatrix[0])      # 获取特征数
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # 获取侮辱性语句的概率 p(c_i)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:   # 若有侮辱性语句
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


'''
  功能：对输入属性进行分类
  输入：（待分类属性）（类别0条件概率log）（类别1条件概率log）（类别1的概率）
  输出：样本所属类别
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(pClass1)
    if p1>p0:
        return 1
    else:
        return 0

'''
功能：测试贝叶斯分类器
'''
def testingNB():
    listOposts, listClasses = loadDataSet()     # 获取训练数据
    myVocabSet = createVocabList(listOposts)    # 获取词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabSet, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))   # 获取先验概率和类条件概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabSet, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabSet, testEntry))
    print(testEntry, ' classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

# ----------------------------------------------------------------------------------------------------------------------
# 邮件分类

'''
功能：接受一个大字符串并将其解析为字符串列表。
该函数去掉少于两个字符的字符串，并将所有字符串转为小写
'''
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

'''
功能：构造邮件分类器并随机选择10个样本进行测试
'''
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding='ANSI').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding='ANSI').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        ranIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[ranIndex])
        del trainingSet[ranIndex]
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount/len(testSet)))

# ---------------------------------------------------------------------------------------------------------------------
# 使用朴素贝叶斯分类器从个人广告中获取区域倾向



'''
功能：遍历词汇表中每个词并统计它在文本中出现的次数，根据次数从高到低对词典进行排序
'''
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        pass
# -----------------------------------------未完

if __name__ == '__main__':
    # listOPosts, listClasses = loadDataSet()
    # myVocabSet = createVocabList(listOPosts)
    # print(len(myVocabSet))
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabSet, postinDoc))
    # p0V,p1V,pAb = trainNB0(array(trainMat), array(listClasses))
    # print(p0V,'\n',p1V,'\n',pAb)
    # testingNB()
    spamTest()
