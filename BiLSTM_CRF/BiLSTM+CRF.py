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

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from data_processor import Mydataset
from model import BiLSTM_CRF

torch.manual_seed(1)

embedding_size = 128
hidden_dim = 384
epochs = 1
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
model = BiLSTM_CRF(len(train_dataset.vocab), train_dataset.label_map, embedding_size, hidden_dim, device).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

def train():
    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        losses = 0
        step = 1
        epoch_start = time.time()
        for text, label, seq_len in train_dataloader:
            start = time.time()
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)

            loss = model.neg_log_likelihood(text, label, seq_len)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch: [{epoch + 1}/{epochs}], '
                  f'finished: {step * batch_size / len(train_dataset) * 100:<2.2f}%, '
                  f'loss: {loss.item():<2.4f}, '
                  f'time: {time.time() - start:<2.2f}s, '
                  f'remaining time: {(len(train_dataset) / batch_size - step) / step * (time.time() - epoch_start):<.0f}s')
            step += 1

            torch.save(model.state_dict(), './model.bin')

def evaluate():
    model.load_state_dict(torch.load('./model.bin'))
    all_label = []
    all_pred = []
    for text, label, seq_len in valid_dataloader:
        text = text.to(device)
        seq_len = seq_len.to(device)
        a = label[0][:seq_len[0]]
        all_label.extend([l for i, batch in enumerate(label) for l in batch[:seq_len[i]].tolist()])
        batch_tag = model.get_labels(text, seq_len)
        print(np.mean(batch_tag))
        all_label.extend([l[:i].items() for i, l in enumerate(label)])
        all_pred.extend(batch_tag)

evaluate()