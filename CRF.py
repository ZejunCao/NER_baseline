#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2022/10/6 20:27
# @File    : CRF.py
# @Software: PyCharm
# @description: 使用单个CRF进行命名实体识别NER

# 注：本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.
# 数据集详情介绍：https://www.cluebenchmarks.com/introduce.html
# 数据集下载链接：https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip
# 代码参考：https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

'''
    该NER任务使用BIO三位标注法，即:
    B-begin：代表实体开头
    I-inside：代表实体内部
    O-outside：代表不属于任何实体

    其后面接实体类型，如'B-name','I-company'
'''


import json
import sklearn_crfsuite
from sklearn_crfsuite import metrics


# 将数据处理成CRF库输入格式
def data_process(path):
    # 读取每一条json数据放入列表中
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    json_data = []
    with open(path, 'r') as fp:
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


# 判断字符是否是英文
def is_english(c):
    if ord(c.lower()) >= 97 and ord(c.lower()) <= 122:
        return True
    else:
        return False

# 将文本转换为特征字典
# sklearn-crfsuite输入数据支持多种格式，这里选择字典格式
# 单个CRF与BiLSTM+CRF不同，BiLSTM会自动生成输入序列中每个字符的发射概率，而单个CRF的发射概率则是通过学习将特征映射成发射概率
# sklearn-crfsuite的数据输入格式采用字典格式，类似于做特征工程，CRF将这些特征映射成发射概率
'''
    序列中的每一个字符处理成如下格式：
    {'bias': 1.0,
     'word': '商',
     'word.isdigit()': False,
     'word.is_english()': False,
     '-1:word': '浙',
     '-1:word.isdigit()': False,
     '-1:word.is_english()': False,
     '+1:word': '银',
     '+1:word.isdigit()': False,
     '+1:word.is_english()': False}
'''
def word2features(sent, i):
    # 本代码采用大小为3的滑动窗口构造特征，特征有当前字符、字符是否为数字或英文等，当然可以增大窗口或增加其他特征
    # 特征长度可以不同
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit(),
        'word.is_english()': is_english(word),
    }

    if i > 0:
        word = sent[i - 1][0]
        features.update({
            '-1:word': word,
            '-1:word.isdigit()': word.isdigit(),
            '-1:word.is_english()': is_english(word),
        })
    else:
        # 若该字符为序列开头，则增加特征 BOS(begin of sentence)
        features['BOS'] = True
    # 该字的后一个字
    if i < len(sent) - 1:
        word = sent[i + 1][0]
        features.update({
            '+1:word': word,
            '+1:word.isdigit()': word.isdigit(),
            '+1:word.is_english()': is_english(word),
        })
    else:
        # 若该字符为序列结尾，则增加特征 EOS(end of sentence)
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for label in sent]


train = data_process('./data/cluener_public/train.json')
valid = data_process('./data/cluener_public/dev.json')
print('训练集长度:', len(train))
print('验证集长度:', len(valid))
X_train = [sent2features(s[0]) for s in train]
y_train = [sent2labels(s[1]) for s in train]
X_dev = [sent2features(s[0]) for s in valid]
y_dev = [sent2labels(s[1]) for s in valid]
print(X_train[0][1])

# algorithm：lbfgs法求解该最优化问题，c1：L1正则系数，c2：L2正则系数，max_iterations：迭代次数，verbose：是否显示训练信息
crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100,
                                 all_possible_transitions=True, verbose=True)
# 若sklearn版本大于等于0.24会报错：AttributeError: 'CRF' object has no attribute 'keep_tempfiles'
# 可降低版本 pip install -U 'scikit-learn<0.24'
# 或使用异常处理，不会影响训练效果
try:
    crf_model.fit(X_train, y_train)
except:
    pass

labels = list(crf_model.classes_)
# 由于大部分标签都是'O'，故不去关注'O'标签的预测
labels.remove("O")
y_pred = crf_model.predict(X_dev)
# 计算F1分数，average可选'micro'，'macro'，'weighted'，处理多类别F1分数的不同计算方法
# 此metrics为sklearn_crfsuite.metrics，但必须引入from sklearn_crfsuite import metrics
# 也可使用sklearn.metrics.f1_score(y_dev, y_pred, average='weighted', labels=labels)),但要求y_dev和y_pred是一维列表
print('weighted F1 score:', metrics.flat_f1_score(y_dev, y_pred,
                      average='weighted', labels=labels))

# 排好标签顺序输入，否则默认按标签出现顺序进行排列
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
# 打印详细分数报告，包括precision(精确率)，recall(召回率)，f1-score(f1分数)，support(个数)，digits=3代表保留3位小数
print(metrics.flat_classification_report(
    y_dev, y_pred, labels=sorted_labels, digits=3
))

# 查看转移概率和发射概率
# print('CRF转移概率：', crf_model.transition_features_)
# print('CRF发射概率：', crf_model.state_features_)
