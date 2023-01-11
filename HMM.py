#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2022/10/5 15:31
# @File    : HMM.py
# @Software: PyCharm
# @description: 使用HMM进行命名实体识别NER

# 注：本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.
# 数据集详情介绍：https://www.cluebenchmarks.com/introduce.html
# 数据集下载链接：https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip

import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn_crfsuite import metrics

# 读取json数据
json_data = []
with open('./data/cluener_public/train.json', 'r', encoding='utf-8') as fp:
    for line in fp:
        json_data.append(json.loads(line))

'''
json_data[0]数据为该格式：

{'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
 'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
'''

# 统计共有多少类别
n_classes = []
for data in json_data:
    for i in data['label'].keys():  # 获取实体标签，如'name'，'company'
        if i not in n_classes:  # 将新的标签加入到列表中
            n_classes.append(i)

# n_classes: ['name', 'company', 'game', 'organization', 'movie',
#             'address', 'position', 'government', 'scene', 'book']

# 设计tag2idx字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
tag2idx = defaultdict()
tag2idx['O'] = 0
count = 1
for n_class in n_classes:
    tag2idx['B-' + n_class] = count
    count += 1
    tag2idx['I-' + n_class] = count
    count += 1


def data_process(path):
    # 读取每一条json数据放入列表中
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    json_data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))

    # json_data中每一条数据的格式为
    '''
    {'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
     'label': {'name': {'叶老桂': [[9, 11]]}, 'company': {'浙商银行': [[0, 3]]}}}
     '''

    # 将json文件处理成如下格式
    '''
    [['浙', '商', '银', '行', '企', '业', '信', '贷', '部', '叶', '老', '桂', '博', '士', '则', '从', '另', '一', 
    '个', '角', '度', '对', '五', '道', '门', '槛', '进', '行', '了', '解', '读', '。', '叶', '老', '桂', '认', 
    '为', '，', '对', '目', '前', '国', '内', '商', '业', '银', '行', '而', '言', '，'], 
    ['B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 
    'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
    '''
    data = []
    # 遍历json_data中每组数据
    for i in range(len(json_data)):
        # 将标签全初始化为'O'
        label = ['O'] * len(json_data[i]['text'])
        # 遍历'label'中几组实体，如样例中'name'和'company'
        for n in json_data[i]['label']:
            # 遍历实体中几组文本，如样例中'name'下的'叶老桂'（有多组文本的情况，样例中只有一组）
            for key in json_data[i]['label'][n]:
                # 遍历文本中几组下标，如样例中[[9, 11]]（有时某个文本在该段中出现两次，则会有两组下标）
                for n_list in range(len(json_data[i]['label'][n][key])):
                    # 记录实体开始下标和结尾下标
                    start = json_data[i]['label'][n][key][n_list][0]
                    end = json_data[i]['label'][n][key][n_list][1]
                    # 将开始下标标签设为'B-' + n，如'B-' + 'name'即'B-name'
                    # 其余下标标签设为'I-' + n
                    label[start] = 'B-' + n
                    label[start + 1: end + 1] = ['I-' + n] * (end - start)

        # 对字符串进行字符级分割
        # 英文文本如'bag'分割成'b'，'a'，'g'三位字符，数字文本如'125'分割成'1'，'2'，'5'三位字符
        texts = []
        for t in json_data[i]['text']:
            texts.append(t)

        # 将文本和标签编成一个列表添加到返回数据中
        data.append([texts, label])
    return data


class HMM_model:
    def __init__(self, tag2idx):
        self.tag2idx = tag2idx  # tag2idx字典
        self.n_tag = len(self.tag2idx)  # 标签个数
        self.n_char = 65535  # 所有字符的Unicode编码个数，包括汉字
        self.epsilon = 1e-100  # 无穷小量，防止归一化时分母为0
        self.idx2tag = dict(zip(self.tag2idx.values(), self.tag2idx.keys()))  # idx2tag字典
        self.A = np.zeros((self.n_tag, self.n_tag))  # 状态转移概率矩阵, shape:(21, 21)
        self.B = np.zeros((self.n_tag, self.n_char))  # 观测概率矩阵, shape:(21, 65535)
        self.pi = np.zeros(self.n_tag)  # 初始隐状态概率,shape：(21,)

    def train(self, train_data):
        print('开始训练数据：')
        for i in tqdm(range(len(train_data))):  # 几组数据
            for j in range(len(train_data[i][0])):  # 每组数据中几个字符
                cur_char = train_data[i][0][j]  # 取出当前字符
                cur_tag = train_data[i][1][j]  # 取出当前标签
                self.B[self.tag2idx[cur_tag]][ord(cur_char)] += 1  # 对B矩阵中标签->字符的位置加一
                if j == 0:
                    # 若是文本段的第一个字符，统计pi矩阵
                    self.pi[self.tag2idx[cur_tag]] += 1
                    continue
                pre_tag = train_data[i][1][j - 1]  # 记录前一个字符的标签
                self.A[self.tag2idx[pre_tag]][self.tag2idx[cur_tag]] += 1  # 对A矩阵中前一个标签->当前标签的位置加一

        # 防止数据下溢,对数据进行对数归一化
        self.A[self.A == 0] = self.epsilon
        self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0] = self.epsilon
        self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        self.pi[self.pi == 0] = self.epsilon
        self.pi = np.log(self.pi) - np.log(np.sum(self.pi))

        # 将A，B，pi矩阵保存到本地
        np.savetxt('./A.txt', self.A)
        np.savetxt('./B.txt', self.B)
        np.savetxt('./pi.txt', self.pi)
        print('训练完毕！')

    # 载入A，B，pi矩阵参数
    def load_paramters(self, A='./A.txt', B='./B.txt', pi='./pi.txt'):
        self.A = np.loadtxt(A)
        self.B = np.loadtxt(B)
        self.pi = np.loadtxt(pi)

    # 使用维特比算法进行解码
    def viterbi(self, s):
        # 计算初始概率，pi矩阵+第一个字符对应各标签概率
        delta = self.pi + self.B[:, ord(s[0])]
        # 前向传播记录路径
        path = []
        for i in range(1, len(s)):
            # 广播机制，重复加到A矩阵每一列
            tmp = delta.reshape(-1, 1) + self.A
            # 取最大值作为节点值，并加上B矩阵
            delta = np.max(tmp, axis=0) + self.B[:, ord(s[i])]
            # 记录当前层每一个节点的最大值来自前一层哪个节点
            path.append(np.argmax(tmp, axis=0))

        # 回溯，先找到最后一层概率最大的索引
        index = np.argmax(delta)
        results = [self.idx2tag[index]]
        # 逐层回溯，沿着path找到起点
        while path:
            tmp = path.pop()
            index = tmp[index]
            results.append(self.idx2tag[index])
        # 序列翻转
        results.reverse()
        return results

    def predict(self, s):
        results = self.viterbi(s)
        for i in range(len(s)):
            print(s[i] + results[i], end=' | ')

    def valid(self, valid_data):
        y_pred = []
        # 遍历验证集每一条数据，使用维特比算法得到预测序列，并加到列表中
        for i in range(len(valid_data)):
            y_pred.append(self.viterbi(valid_data[i][0]))
        return y_pred


train_data = data_process('./data/cluener_public/train.json')
valid_data = data_process('./data/cluener_public/dev.json')
print('训练集长度:', len(train_data))
print('验证集长度:', len(valid_data))

model = HMM_model(tag2idx)
model.train(train_data)
model.load_paramters()
model.predict('浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言')
y_pred = model.valid(valid_data)
y_true = [data[1] for data in valid_data]

# 排好标签顺序输入，否则默认按标签出现顺序进行排列
sort_labels = [k for k in tag2idx.keys()]

# 打印详细分数报告，包括precision(精确率)，recall(召回率)，f1-score(f1分数)，support(个数)，digits=3代表保留3位小数
print(metrics.flat_classification_report(
    y_true, y_pred, labels=sort_labels[1:], digits=3
))