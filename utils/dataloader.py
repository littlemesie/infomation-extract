# -*- coding:utf-8 -*-

"""
@date: 2023/5/17 下午5:52
@summary: 数据加载
"""
import json
import torch
import pandas as pd
from torch.utils.data import Dataset

class TextDatasetMulti(Dataset):
    def __init__(self, filepath):
        super(TextDatasetMulti, self).__init__()
        self.texts, self.names, self.label = self.load_data(filepath)

    def load_data(self, path):
        train = pd.read_csv(path)
        train = train[['text', 'name', 'label']].dropna()
        train['label'] = train['label'].apply(lambda x: eval(x))
        texts = train.text.to_list()
        names = train.name.to_list()
        labels = train.label.to_list()
        return texts, names, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        name = self.names[item]
        label = self.label[item]
        return text, name, label

class BatchTextDataset(object):
    def __init__(self, tokenizer, max_len=512, split_len=3):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split_len = split_len

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_name = [item[1] for item in batch]
        batch_label = [item[2] for item in batch]
        batch_token, batch_segment, batch_mask = list(), list(), list()
        for i, text in enumerate(batch_text):
            name = batch_name[i]
            text_split = text.split(name)
            text_split = [t for t in text_split if t]

            token_list = []
            for j, ts in enumerate(text_split):
                if not ts:
                    continue
                if len(ts) > self.max_len-2:
                    ts = ts[:self.max_len-2]
                token = self.tokenizer.tokenize(ts)
                token = ['[CLS]'] + token + ['[SEP]']
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                padding = [0] * (self.max_len - len(token_id))
                token_id = token_id + padding
                token_list.append(token_id)
            if len(token_list) >= self.split_len:
                token_list = token_list[:self.split_len]
            else:
                for k in range(self.split_len - len(token_list)):
                    token_list.append([0] * self.max_len)
            # print(token_list)
            batch_token.append(token_list)
        # print(batch_token)
        batch_tensor_token = torch.tensor(batch_token)
        batch_tensor_label = torch.tensor(batch_label)
        return batch_tensor_token, batch_tensor_label
