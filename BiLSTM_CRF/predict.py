#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/4/7 16:11
# @File    : predict.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 预测文件


import torch

from BiLSTM_CRF.data_processor import get_vocab, get_label_map
from BiLSTM_CRF.model import BiLSTM_CRF

def chunks_extract(pred):
    if not pred:
        return []

    cur_entity = None
    res = []
    st_idx, end_idx = 0, 0
    for i, pred_single in enumerate(pred):
        pred_start_B = pred_single.startswith('B')
        pred_entity = pred_single.split('-')[-1]

        if cur_entity:
            if pred_start_B or cur_entity != pred_entity:
                res.append({
                    'st_idx': st_idx,
                    'end_idx': i,
                    'label': cur_entity
                })
                cur_entity = None
        if pred_start_B:
            st_idx = i
            cur_entity = pred_entity
    if cur_entity:
        res.append({
            'st_idx': st_idx,
            'end_idx': len(pred),
            'label': cur_entity,
        })
    return res

embedding_size = 128
hidden_dim = 768

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 加载训练时保存的词表
vocab = get_vocab()[0]
# 加载训练时保存的标签字典
label_map, label_map_inv = get_label_map()
model = BiLSTM_CRF(embedding_size, hidden_dim, vocab, label_map, device)
model.load_state_dict(torch.load('./model1.bin'))
model.to(device)

text = '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，'

model.eval()
model.state = 'pred'
with torch.no_grad():
    text = [vocab.get(t, vocab['UNK']) for t in text]
    seq_len = torch.tensor(len(text), dtype=torch.long).unsqueeze(0)
    seq_len = seq_len.to(device)
    text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
    text = text.to(device)
    batch_tag = model(text, seq_len)
    pred = [label_map_inv[t] for t in batch_tag]
    print(pred)
    print(chunks_extract(pred))


