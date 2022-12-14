#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/7 23:33
# @File    : data_processor.py
# @Software: PyCharm
# @description: 数据处理，包括label_map处理，dataset建立

import os
import json
import pickle

import torch
from torch.utils.data import Dataset


def get_label_map():
    label_map_path = '../data/cluener_public/label_map.json'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(label_map_path):
        with open(label_map_path, 'r', encoding='utf-8') as fp:
            label_map = json.load(fp)
    else:
        # 读取json数据
        json_data = []
        with open('../data/cluener_public/train.json', 'r', encoding='utf-8') as fp:
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
            for label in data['label'].keys():  # 获取实体标签，如'name'，'company'
                if label not in n_classes:  # 将新的标签加入到列表中
                    n_classes.append(label)
        n_classes.sort()

        # n_classes: ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        # 设计label_map字典，对每个标签设计两种，如B-name、I-name，并设置其ID值
        label_map = {}
        for n_class in n_classes:
            label_map['B-' + n_class] = len(label_map)
            label_map['I-' + n_class] = len(label_map)
        label_map['O'] = len(label_map)
        # 对于BiLSTM+CRF网络，需要增加开始和结束标签，以增强其标签约束能力
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        label_map[START_TAG] = len(label_map)
        label_map[STOP_TAG] = len(label_map)

        # 将label_map字典存储成json文件
        with open(label_map_path, 'w', encoding='utf-8') as fp:
            json.dump(label_map, fp, indent=4)
    label_map_inv = {v: k for k, v in label_map.items()}
    return label_map, label_map_inv

def get_vocab():
    vocab_path = '../data/cluener_public/vocab.pkl'
    # 第一次运行需要遍历训练集获取到标签字典，并存储成json文件保存，第二次运行即可直接载入json文件
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as fp:
            vocab = pickle.load(fp)
    else:
        json_data = []
        with open('../data/cluener_public/train.json', 'r', encoding='utf-8') as fp:
            for line in fp:
                json_data.append(json.loads(line))
        vocab = {'PAD': 0, 'UNK': 1}
        for data in json_data:
            for word in data['text']:  # 获取实体标签，如'name'，'compan
                if word not in vocab:
                    vocab[word] = len(vocab)
        with open(vocab_path, 'wb') as fp:
            pickle.dump(vocab, fp)

    vocab_inv = {v: k for k, v in vocab.items()}
    return vocab, vocab_inv


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


class Mydataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = data_process(self.file_path)
        self.label_map, self.label_map_inv = get_label_map()
        # 建立中文词表，扫描训练集所有字符得到，'PAD'在batch填充时使用，'UNK'用于替换字表以外的新字符
        self.vocab, self.vocab_inv = get_vocab()
        self.examples = []
        for text, label in self.data:
            t = [self.vocab.get(t, self.vocab['UNK']) for t in text]
            l = [self.label_map[l] for l in label]
            self.examples.append([t, l])
        self.a = 0

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.data)

    def collect_fn(self, batch):
        text = [t for t, l in batch]
        label = [l for t, l in batch]
        seq_len = [len(i) for i in text]
        max_len = max(seq_len)

        text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
        label = [l + [self.label_map['O']] * (max_len - len(l)) for l in label]

        text = torch.LongTensor(text)
        label = torch.LongTensor(label)
        seq_len = torch.LongTensor(seq_len)

        return text, label, seq_len