from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据
X = []
Y = []
fr = open("datingTestSet.txt")
index = 0
for line in fr.readlines():
    line = line.strip()
    line = line.split('\t')
    X.append(line[:3])
    Y.append(line[-1])

#归一化
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 交叉分类
train_X,test_X, train_y, test_y = train_test_split(X,
                                                   Y,
                                                   test_size=0.2) # test_size:测试集比例20%

# KNN模型，选择3个邻居
model = GaussianNB()

model.fit(train_X, train_y)
print(model)

expected = test_y
predicted = model.predict(test_X)
print(metrics.classification_report(expected, predicted))       # 输出分类信息
label = list(set(Y))    # 去重复，得到标签类别
print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵信息
