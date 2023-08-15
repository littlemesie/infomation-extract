# -*- coding:utf-8 -*-

"""
@date: 2023/8/3 下午7:26
@summary:
"""
from transformers import BertTokenizer
# 加载分词器
def load_tokenizer(model_path, special_token=None):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer