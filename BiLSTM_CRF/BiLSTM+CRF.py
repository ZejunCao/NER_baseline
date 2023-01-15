#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/7 20:20
# @File    : BiLSTM+CRF.py
# @Software: PyCharm
# @description: 使用BiLSTM+CRF进行命名实体识别NER

# 注：本数据是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS.
# 数据集详情介绍：https://www.cluebenchmarks.com/introduce.html
# 数据集下载链接：https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip
# 代码参考：https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

import time
import datetime
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
from itertools import chain

from BiLSTM_CRF.data_processor import Mydataset
from BiLSTM_CRF.model import BiLSTM_CRF

# 设置torch随机种子
torch.manual_seed(3407)

embedding_size = 128
hidden_dim = 768
epochs = 50
batch_size = 32

device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_dataset = Mydataset('../data/cluener_public/train.json')
valid_dataset = Mydataset('../data/cluener_public/dev.json')
print('训练集长度:', len(train_dataset))
print('验证集长度:', len(valid_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                              collate_fn=train_dataset.collect_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=False,
                              collate_fn=train_dataset.collect_fn)
model = BiLSTM_CRF(train_dataset, embedding_size, hidden_dim, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def train():
    total_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        for step, (text, label, seq_len) in enumerate(train_dataloader, start=1):
            start = time.time()
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)

            loss = model(text, label, seq_len)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f'Epoch: [{epoch + 1}/{epochs}],  '
                  f'cur epoch finished: {step * batch_size / len(train_dataset) * 100:<2.2f}%,  '
                  f'loss: {loss.item():<2.4f},  '
                  f'time: {time.time() - start:<2.2f}s,  '
                  f'cur epoch remaining time: {datetime.timedelta(seconds=int((len(train_dataloader) - step) / step * (time.time() - epoch_start)))}',
                  f'total remaining time: {datetime.timedelta(seconds=int((len(train_dataloader) - step) * epochs / step * (time.time() - total_start)))}')

            torch.save(model.state_dict(), './model1.bin')
        evaluate()

def evaluate():
    # model.load_state_dict(torch.load('./model1.bin'))
    all_label = []
    all_pred = []
    model.eval()
    model.state = 'eval'
    for text, label, seq_len in tqdm(valid_dataloader, desc='eval: '):
        text = text.to(device)
        seq_len = seq_len.to(device)
        batch_tag = model(text, label, seq_len)
        all_label.extend([[train_dataset.label_map_inv[t] for t in l[:seq_len[i]].tolist()] for i, l in enumerate(label)])
        all_pred.extend([[train_dataset.label_map_inv[t] for t in l] for l in batch_tag])

    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    sort_labels = [k for k in train_dataset.label_map.keys()]
    print(metrics.classification_report(
        all_label, all_pred, labels=sort_labels[:-3], digits=3
    ))

train()

def date_norm(second: int) -> str:
    """
    将time.time()做差得到的秒数转化成时分秒
    :param second: int 秒数
    :return: str 时分秒
    """
    if second < 60:
        return str(second)
    else:
        second /= 60
        return date_norm(second)
