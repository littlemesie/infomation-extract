# -*- coding:utf-8 -*-

"""
@date: 2023/8/3 下午6:20
@summary: bert bilstm crf 实体提取
"""
import os
import sys
import time
import torch
import numpy as np
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataloader import DatasetNER
from utils.dataloader import BertNERBatchDataset
from utils.time_util import get_time_diff
from utils.load_util import load_tokenizer
from models.relation_extract.text_cnn_classfication import TextCNNRE

tag2ids = {
    'O': 0,
    'B-name': 1, 'I-name': 2, 'E-name': 3,
    'B-district': 4, 'I-district': 5, 'E-district': 6,
    'B-hotel': 7, 'I-hotel': 8, 'E-hotel': 9,
    'B-ktv': 10, 'I-ktv': 11, 'E-ktv': 12,
    'B-restaurant': 13, 'I-restaurant': 14, 'E-restaurant': 15,
    'B-school': 16, 'I-school': 17, 'E-school': 18,
    'B-store': 19, 'I-store': 20, 'E-store': 21,
    'B-clinic': 22, 'I-clinic': 23, 'E-clinic': 24,
    'B-drugstore': 25, 'I-drugstore': 26, 'E-drugstore': 27,
    'B-hospital': 28, 'I-hospital': 29, 'E-hospital': 30,
    'S-name': 31, 'S-district': 32, 'S-hotel': 33,  'S-ktv': 34, 'S-restaurant': 35,  'S-school': 36, 'S-store': 37,
    'S-clinic': 38, 'S-drugstore': 39, 'S-hospital': 40
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model_path = "/home/mesie/python/aia-nlp-service/lib/pretrained/albert_chinese_base"


def get_train_dataloader(tokenizer):
    dataset_batch = BertNERBatchDataset(tokenizer, tag2ids, max_seq_len=512)

    train_dataset = DatasetNER(f"../data/relation/train_data.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    valid_dataset = DatasetNER(f"../data/relation/valid_data.csv")
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    return train_dataloader, valid_dataloader
