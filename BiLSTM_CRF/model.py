#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/8 0:31
# @File    : model.py
# @Software: PyCharm
# @description: 模型文件

import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq):
    idxs = [ord(w) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    a = len(vec.size())
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, label_map, embedding_dim, hidden_dim, device='cpu'):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label_map = label_map
        self.tagset_size = len(label_map)
        self.device = device

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[label_map[START_TAG], :] = -10000
        self.transitions.data[:, label_map[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats, seq_len):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((self.tagset_size,), -10000.)
        # START_TAG has all of the score.
        init_alphas[self.label_map[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32, device=self.device)
        forward_var[:, 0, :] = init_alphas

        transitions = self.transitions.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        # Iterate through the sentence
        for seq_i in range(feats.shape[1]):
            emit_score = feats[:, seq_i, :]
            tag_var = (
                    forward_var[:, seq_i, :].unsqueeze(1).repeat(1, feats.shape[2], 1)  # (32, 23, 23)
                    + transitions
                    + emit_score.unsqueeze(2).repeat(1, 1, feats.shape[2])
            )
            # 这里必须调用clone，不能直接在forward_var上修改，否则在梯度回传时会报错
            cloned = forward_var.clone()
            cloned[:, seq_i + 1, :] = log_sum_exp(tag_var)
            forward_var = cloned
        forward_var = forward_var[range(feats.shape[0]), seq_len, :]
        terminal_var = forward_var + self.transitions[self.label_map[STOP_TAG]].unsqueeze(0).repeat(feats.shape[0], 1)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, seq_len):
        # Gives the score of a provided tag sequence
        score = torch.zeros(feats.shape[0], device=self.device)
        start = torch.tensor([self.label_map[START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0], 1)
        tags = torch.cat([start, tags], dim=1)
        for batch_i in range(feats.shape[0]):
            for seq_i in range(seq_len[batch_i]):
                score[batch_i] += self.transitions[tags[batch_i][seq_i + 1], tags[batch_i][seq_i]] \
                                 + feats[batch_i][seq_i][tags[batch_i][seq_i + 1]]
            score[batch_i] += self.transitions[self.label_map[STOP_TAG], tags[batch_i][seq_len[batch_i]]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.label_map[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.label_map[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.label_map[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def _get_lstm_features(self, sentence, seq_len):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(seq_unpacked)
        return lstm_feats

    def neg_log_likelihood(self, sentence, tags, seq_len):
        feats = self._get_lstm_features(sentence, seq_len)
        forward_score = self._forward_alg(feats, seq_len)
        gold_score = self._score_sentence(feats, tags, seq_len)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def get_labels(self, sentence, seq_len):
        feats = self._get_lstm_features(sentence, seq_len)
        all_tag = []
        for i, feat in enumerate(feats):
            path_score, best_path = self._viterbi_decode(feat[:seq_len[i]])
            all_tag.extend(best_path)
        return all_tag
