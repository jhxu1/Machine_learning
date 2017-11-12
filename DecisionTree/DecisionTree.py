from math import log
import operator
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']


class DecisionTree:
    def __init__(self):
        self.__dataset = []
        self.__label = []
        self.__tree = {}

    # 计算信息熵Ent(D)
    @staticmethod
    def __cal_shanno_ent(dataset):
        num = len(dataset)
        label_count = {}
        for sample in dataset:  # 计算样本中每一类个数，存入label_count
            label = sample[-1]
            if label not in label_count.keys():
                label_count[label] = 0
            label_count[label] += 1
        shanno_ent = 0.0
        for key in label_count:  # 计算香农熵
            prob = float(label_count[key]) / num
            shanno_ent -= prob * log(prob, 2)
        return shanno_ent

    # 读取数据
    def load_dataset(self, filename):
        self.__dataset.clear()
        with open(filename) as f:
            line = f.readline()
            if line:
                self.__label = line.split(',')
                line = f.readline()
                while line:
                    self.__dataset.append(line.split(','))
                    line = f.readline()

    '''
    功能：生成一棵树
    '''
    def create_tree(self):
        self.__tree = DecisionTree.__create_tree(self.__dataset, self.__label)
        print(self.__tree)

    '''
    功能：在给定数据集合中寻找满足条件（属性吻合）的子集合
    输入：dataset（待分类数据集） axis（划分属性索引） value（划分属性取值）
    输出：ret_dataset（子数据集）
    '''
    @staticmethod
    def __get_split_dataset(dataset, axis, value):
        ret_dataset = []
        for feat_vec in dataset:
            if feat_vec[axis] == value:
                reduce_feat_vec = feat_vec[:axis]
                reduce_feat_vec.extend(feat_vec[axis+1:])
                ret_dataset.append(reduce_feat_vec)
        return ret_dataset

    '''
    功能：选择当前数据集中的最佳分类
    输入：dataset（数据集）
    输出：best_feature（最优属性索引）
    '''
    @staticmethod
    def __choose_best_feature_to_split(dataset):
        num_feature = len(dataset[0])
        base_entropy = DecisionTree.__cal_shanno_ent(dataset)   # dataset的信息熵
        best_info_gain = 0.0                                    # 最大信息增益
        best_feature = -1
        for i in range(num_feature-1):
            feat_list = [example[i] for example in dataset]
            unique_vals = set(feat_list)                        # 获取第i种属性的种类集合
            new_entropy = 0.0
            for value in unique_vals:
                sub_dataset = DecisionTree.__get_split_dataset(dataset, i, value)
                prob = len(sub_dataset)/float(len(dataset))
                new_entropy += prob * DecisionTree.__cal_shanno_ent(sub_dataset)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_feature = i
                best_info_gain = info_gain
        if feat_list == -1:
            print("--------------wrong----------------")
        return best_feature

    '''
    功能：找出一个队列中数量最多的数
    输入：classlist（输入队列）
    输出：数量最多的数
    '''
    @staticmethod
    def __majority_cnt(classlist):
        class_count = {}
        for vote in classlist:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        print(class_count.items())
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    '''
    功能：使用嵌套字典生成一棵决策树
    输入：dateset（数据集） labels（数据集属性名）
    输出：tree（嵌套dict 决策树）
    '''
    @staticmethod
    def __create_tree(dataset, labels):
        class_list = [example[-1] for example in dataset]
        data_list = [example[:-1] for example in dataset]
        if class_list.count(class_list[0]) == len(class_list):       # 若全属于同一类别
            return class_list[0]
        if len(dataset) == 1 or data_list.count(data_list[0]) == len(data_list):           # 若属性集合为空集
            return DecisionTree.__majority_cnt(class_list)
        best_feat = DecisionTree.__choose_best_feature_to_split(dataset)        # 获取最优属性索引
        best_feat_label = labels[best_feat]
        mytree = {best_feat_label: {}}
        sub_label = labels[:]                                      # 删除label最优属性列表
        del(sub_label[best_feat])
        feat_value = [example[best_feat] for example in dataset]    # 获取a_*中的每一项a_*^v
        unique_value = set(feat_value)
        for value in unique_value:
            sub_dataset = DecisionTree.__get_split_dataset(dataset, best_feat, value)
            mytree[best_feat_label][value] = DecisionTree.__create_tree(sub_dataset, sub_label)
        return mytree

    def plot_tree(self):
        create_plot(self.__tree)


decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


'''
功能：获取嵌套dict叶子节点的数目
输入：mytree（嵌套dict）
'''
def get_num_leafs(mytree):
    num_leafs = 0
    first_side = list(mytree.keys())
    first_str = first_side[0]
    second_dict = mytree[first_str]
    for key in second_dict:
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

'''
功能：获取嵌套字典dict树的层数
输入：mytree（dict）
'''
def get_tree_depth(mytree):
    maxdepth = 0
    first_str = list(mytree.keys())[0]
    second_dict = mytree[first_str]
    for key in second_dict:
        a = type(second_dict[key]).__name__
        if type(second_dict[key]).__name__ == 'dict':  # 如果还有子节点
            thisdepth = 1 + get_tree_depth(second_dict[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth:
            maxdepth = thisdepth
    return maxdepth


'''
功能：绘制从parentpt指向centerpt的箭头
输入：nodetxt（centerpt处需要的文字） centerpt（目标坐标） parentpt（起点坐标） nodetype（箭头形式）
'''
def plot_node(nodetxt, centerpt, parentpt, nodetype):
    create_plot.axl.annotate(nodetxt, xy=parentpt, xycoords='axes fraction',
                                          xytext=centerpt, textcoords='axes fraction',
                                          va="center", ha="center", bbox=nodetype, arrowprops=arrow_args)


'''
功能：在父子节点间填充文本信息
'''
def plot_mid_tex(cntrpt, parentpt, txtstring):
    x_mid = (parentpt[0] - cntrpt[0])/2.0 + cntrpt[0]
    y_mid = (parentpt[1] - cntrpt[1])/2.0 + cntrpt[1]
    create_plot.axl.text(x_mid, y_mid, txtstring)


def plot_tree(mytree, parentpt, nodetxt):
    num_leafs = get_num_leafs(mytree)
    depth = get_tree_depth(mytree)
    first_str = list(mytree.keys())[0]
    cntrpt = (plot_tree.xoff + (1.0 + float(num_leafs))/2.0/plot_tree.totalW,
              plot_tree.yoff)
    plot_mid_tex(cntrpt, parentpt, nodetxt)
    plot_node(first_str, cntrpt, parentpt, decision_node)
    second_dict = mytree[first_str]
    plot_tree.yoff = plot_tree.yoff -1.0/plot_tree.totalD
    for key in second_dict.keys():
        a = second_dict[key]
        if type(second_dict[key]).__name__ == 'dict':
             plot_tree(second_dict[key], cntrpt, str(key))
        else:
            plot_tree.xoff = plot_tree.xoff + 1.0/plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xoff, plot_tree.yoff), cntrpt, leaf_node)
            plot_mid_tex((plot_tree.xoff, plot_tree.yoff), cntrpt, str(key))
    plot_tree.yoff = plot_tree.yoff + 1.0/plot_tree.totalD


def create_plot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.axl = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(intree))
    plot_tree.totalD = float(get_tree_depth(intree))
    plot_tree.xoff = -0.5/plot_tree.totalW
    plot_tree.yoff = 1.0
    plot_tree(intree, (0.5, 1.0), '')
    plt.show()