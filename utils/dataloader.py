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


class DatasetRelation(Dataset):
    def __init__(self, filepath):
        super(DatasetRelation, self).__init__()
        self.texts, self.entity1_list, self.entity2_list, self.relations = self.load_data(filepath)

    def load_data(self, path):
        train = pd.read_csv(path)
        train = train[['text', 'entity1', 'entity2', 'relation']].dropna()
        train['entity1'] = train['entity1'].apply(lambda x: eval(x))
        train['entity2'] = train['entity2'].apply(lambda x: eval(x))
        texts = train.text.to_list()
        entity1_list = train.entity1.to_list()
        entity2_list = train.entity2.to_list()
        relations = train.relation.to_list()
        return texts, entity1_list, entity2_list, relations

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        entity1 = self.entity1_list[item]
        entity2 = self.entity2_list[item]
        relations = self.relations[item]
        return text, entity1, entity2, relations


class RelationBatchDataset(object):
    def __init__(self, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def convert_pos_to_mask(self, e_pos):
        """mask entity pos"""
        e_pos_mask = [0] * self.max_len
        for i in range(len(e_pos)-1):
            if e_pos[i] < self.max_len and e_pos[i+1] < self.max_len:
                for j in range(e_pos[i], e_pos[i+1]):
                    e_pos_mask[j] = 1
        return e_pos_mask

    def __call__(self, batch):
        batch_text = [item[0] for item in batch]
        batch_e1 = [item[1] for item in batch]
        batch_e2 = [item[2] for item in batch]
        batch_relation = [item[3] for item in batch]
        batch_token_ids, batch_token_type_ids, batch_mask, batch_e1_mask, batch_e2_mask = [], [], [], [], []
        for i, text in enumerate(batch_text):
            word_list = list(text)
            e1_mask_list = self.convert_pos_to_mask(batch_e1[i]['pos'])
            e2_mask_list = self.convert_pos_to_mask(batch_e2[i]['pos'])
            batch_e1_mask.append(e1_mask_list)
            batch_e2_mask.append(e2_mask_list)
            for k, p in enumerate(batch_e1[i]['pos']):
                if k % 2 == 0:
                    word_list.insert(p, '[unused1]')
                else:
                    word_list.insert(p, '[unused3]')
            for k, p in enumerate(batch_e2[i]['pos']):
                if k % 2 == 0:
                    word_list.insert(p, '[unused2]')
                else:
                    word_list.insert(p, '[unused4]')

                word_list = word_list[:self.max_len - 2]
            # token = self.tokenizer.tokenize(text)
            token = ['[CLS]'] + word_list + ['[SEP]']
            encoded = self.tokenizer.encode_plus(token, max_length=self.max_len, pad_to_max_length=True)
            batch_token_ids.append(encoded['input_ids'])
            batch_token_type_ids.append(encoded['token_type_ids'])
            batch_mask.append(encoded['attention_mask'])

        # print(batch_token)
        batch_data = {
            'token_ids': torch.tensor(batch_token_ids),
            'token_type_ids': torch.tensor(batch_token_type_ids),
            'attention_mask': torch.tensor(batch_mask),
            'e1_mask': torch.tensor(batch_e1_mask),
            'e2_mask': torch.tensor(batch_e2_mask),
            'relation': torch.tensor(batch_relation)
        }

        return batch_data