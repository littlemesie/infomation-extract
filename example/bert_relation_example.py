# -*- coding:utf-8 -*-

"""
@date: 2023/6/19 上午9:48
@summary: 基于bert分类的关系抽取
"""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from tqdm import tqdm
from sklearn import metrics
from datetime import timedelta
from transformers import BertTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataloader import DatasetRelation
from utils.dataloader import RelationBatchDataset
from models.relation_extract.bert_classfication import BertRE

relation_list = ['unknown', '祖孙', '同门', '上下级', '师生', '情侣', '亲戚', '夫妻', '好友', '兄弟姐妹', '父母', '合作']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model_path = "/home/mesie/python/aia-nlp-service/lib/pretrained/albert_chinese_base"

def get_time_diff(start_time):
    # 获取已使用时间
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 加载分词器
def load_tokenizer(model_path, special_token=None):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def get_train_dataloader(tokenizer):
    dataset_batch = RelationBatchDataset(tokenizer, max_len=256)

    train_dataset = DatasetRelation(f"../data/relation/train_data.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    valid_dataset = DatasetRelation(f"../data/relation/valid_data.csv")
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    return train_dataloader, valid_dataloader

# 模型评估
def evaluation(model, valid_dataloader, criterion):
    model.eval()
    total_loss = 0
    pred_list = np.array([], dtype=int)
    label_list = np.array([], dtype=int)
    for ind, batch_data in enumerate(valid_dataloader):
        token_ids = batch_data['token_ids'].to(device)
        token_type_ids = batch_data['token_type_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        e1_mask = batch_data['e1_mask'].to(device)
        e2_mask = batch_data['e2_mask'].to(device)
        label = batch_data['relation'].to(device)

        out = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
        loss = criterion(out, label)
        total_loss += loss.detach().item()

        label = label.data.cpu().numpy()
        pred = torch.max(out.data.cpu(), 1)[1].numpy()
        label_list = np.append(label_list, label)
        pred_list = np.append(pred_list, pred)

    acc = metrics.accuracy_score(label_list, pred_list)

    return acc, total_loss / len(valid_dataloader)


def train(tokenizer, load_model=False):
    start_time = time.time()
    train_dataloader, valid_dataloader = get_train_dataloader(tokenizer)
    emb_size, num_classes = 768, len(relation_list)
    model = BertRE(emb_size, num_classes, pretrained_model_path)

    model_dir = f"../model_file/relation"
    best_model_path = f"{model_dir}/bert_relation_best.pt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if load_model:
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.01)
    criterion = F.cross_entropy

    loss_total, top_acc = [], 0
    valid_best_loss = float('inf')
    last_improve = 0
    total_batch = 0
    stop_flag = False
    train_acc_history, train_loss_history = [], []
    valid_acc_history, valid_loss_history = [], []
    epochs = 1
    for epoch in range(epochs):
        print("epoch [{}/{}]".format(epoch + 1, epochs))
        model.train()
        epoch_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        for i, batch_data in enumerate(tqdm_bar):
            token_ids = batch_data['token_ids'].to(device)
            token_type_ids = batch_data['token_type_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            e1_mask = batch_data['e1_mask'].to(device)
            e2_mask = batch_data['e2_mask'].to(device)
            labels = batch_data['relation'].to(device)
            model.zero_grad()
            out = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 500 == 0 and total_batch != 0:
                valid_acc, valid_loss = evaluation(model, valid_dataloader, criterion)
                true_label = labels.data.cpu().numpy()
                # pred_label = out.data.numpy()
                pred_label = torch.max(out.data.cpu(), 1)[1].numpy()


                train_acc = metrics.accuracy_score(true_label, pred_label)

                train_acc_history.append(train_acc)
                train_loss_history.append(loss.detach().item())
                valid_acc_history.append(valid_acc)
                valid_loss_history.append(valid_loss)

                if epoch and epoch % 5 == 0:
                    save_model_path = f"{model_dir}/bert_relation_{epoch}.pt"
                    torch.save(model.state_dict(), save_model_path)

                # evaluate on validate data
                if valid_loss < valid_best_loss:
                    valid_best_loss = valid_loss
                    # 保存最好好的模型

                    torch.save(model.state_dict(), best_model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ""
                time_diff = get_time_diff(epoch_time)
                msg = "Iter：{0:6}, Train_Loss: {1:>5.2}, Train_Acc: {2:>6.2%}, Val_Loss: {3:5.2}, Val_Acc: {4:6.2%}, Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, valid_loss, valid_acc, time_diff, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > 1500:
                print('No optimization for a long time, auto-stopping......')
                stop_flag = True
                break
        # 最后一轮保存模型
        if epoch == epochs - 1:
            torch.save(model.state_dict(), best_model_path)
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
        if stop_flag:
            break
        time.sleep(0.5)

def convert_pos_to_mask(e_pos, max_len):
    """mask entity pos"""
    e_pos_mask = [0] * max_len
    for i in range(len(e_pos)-1):
        if e_pos[i] < max_len and e_pos[i+1] < max_len:
            for j in range(e_pos[i], e_pos[i+1]):
                e_pos_mask[j] = 1
    return e_pos_mask

def calc_pos(text, entity):
    s_pos = [i for i in range(len(text)) if text.startswith(entity, i)]
    pos = []
    for p in s_pos:
        pos.append(p)
        pos.append(p + len(entity))
    return pos


def predict(text, entity1, entity2, tokenizer):
    """"""
    emb_size, num_classes, max_len = 768, len(relation_list), 128
    model = BertRE(emb_size, num_classes, pretrained_model_path).to(device)
    best_model_path = f"../model_file/relation/bert_relation_best.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    import time
    # t1 = time.time()
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        word_list = list(text)
        e1_pos = calc_pos(text, entity1)
        e2_pos = calc_pos(text, entity2)
        e1_mask = convert_pos_to_mask(e1_pos, max_len)
        e2_mask = convert_pos_to_mask(e2_pos, max_len)

        for k, p in enumerate(e1_pos):

            if k % 2 == 0:
                word_list.insert(p, '[unused1]')
            else:
                word_list.insert(p, '[unused3]')
        for k, p in enumerate(e2_pos):
            if k % 2 == 0:
                word_list.insert(p, '[unused2]')
            else:
                word_list.insert(p, '[unused4]')

        word_list = word_list[:max_len - 2]
        # token = self.tokenizer.tokenize(text)
        token = ['[CLS]'] + word_list + ['[SEP]']
        encoded = tokenizer.encode_plus(token, max_length=max_len, pad_to_max_length=True)
        token_ids = torch.tensor(encoded['input_ids']).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoded['token_type_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0).to(device)
        e1_mask = torch.tensor(e1_mask).unsqueeze(0).to(device)
        e2_mask = torch.tensor(e2_mask).unsqueeze(0).to(device)
        out = model(token_ids, token_type_ids, attention_mask, e1_mask, e2_mask)
        pred_relation = torch.max(out.data, 1)[1].to(device).numpy()[0]
        pred_relation = relation_list[pred_relation]
        print(pred_relation)
    print(time.time() - t1)

if __name__ == '__main__':
    """"""

    tokenizer = load_tokenizer(pretrained_model_path)
    # train(tokenizer)
    text, entity1, entity2 = "-婚姻2000年1月，冯钰棋与王学兵在一个朋友的派对上相识，9个月后秘密举行婚礼。", "冯钰棋", "王学兵"
    predict(text, entity1, entity2, tokenizer)



