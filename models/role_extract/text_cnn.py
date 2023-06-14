# -*- coding:utf-8 -*-

"""
@date: 2023/6/6 下午2:18
@summary:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, num_kernels, kernel_size, stride=1, emb_size=128,
                 dropout=0.0, padding_index=0):
        super(TextCNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_index)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_kernels, (k, emb_size*3), stride) for k in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_kernels * len(kernel_size), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text_list):
        cat_pool = None
        for i, text in enumerate(text_list):
            emb = self.emb(text)
            emb = torch.reshape(emb, [1, emb.shape[1], emb.shape[0] * emb.shape[2]])
            emb = emb.unsqueeze(1)
            pool = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs], 1)
            pool = self.dropout(pool)
            if i == 0:
                cat_pool = pool
            else:
                cat_pool = torch.cat([cat_pool, pool], dim=0)

        out = self.fc(cat_pool)
        return out