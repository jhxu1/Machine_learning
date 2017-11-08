from math import log
import DecisionTree as DT



def cal_shanno_ent(dataset):
    num = len(dataset)
    label_count = {}
    for sample in dataset:                                  # 计算样本中每一类个数，存入label_count
        label = sample[-1]
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
    shanno_ent = 0.0
    for key in label_count:                                 # 计算香农熵
        prob = float(label_count[key])/num
        shanno_ent -= prob * log(prob, 2)
    return shanno_ent


def create_dataset():
    date_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['np surfacing', 'flippers']
    return date_set, labels


if __name__ == '__main__':
    dt = DT.DecisionTree()
    dt.load_dataset('test.csv')
    dt.create_tree()
    print()

