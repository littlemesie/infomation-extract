# -*- coding:utf-8 -*-

"""
@date: 2023/10/18 下午5:45
@summary:
"""
import torch
import torch.nn as nn
from torchcrf import CRF
import torch.nn.functional as F
from models.ner_extract.model_output import ModelOutput



class BilstmCRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_labels, num_layers=1, dropout=0.5):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            num_labels:标注的种类
            num_layers: rnn层数
            dropout: dropout
        """
        super(BilstmCRF, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                              dropout=dropout)
        self.linear = nn.Linear(2 * hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, texts, labels=None):
        # print(texts.shape)
        emb = self.embedding(texts)
        # print(emb.shape)
        batch_size = emb.size(0)
        max_seq_len = emb.size(1)
        seq_out, _ = self.bilstm(emb)
        seq_out = seq_out.contiguous().view(-1, self.hidden_size * 2)
        seq_out = seq_out.contiguous().view(batch_size, max_seq_len, -1)  # [batch_size, max_seq_len, num_tags]
        seq_out = self.linear(seq_out)
        logits = self.crf.decode(seq_out)
        loss = None
        if labels is not None:
            loss = -self.crf(seq_out, labels, reduction='mean')

        if labels is None:
            return ModelOutput(logits, labels, loss, seq_out)
        out = ModelOutput(logits, labels, loss)
        return out
