# -*- coding:utf-8 -*-

"""
@date: 2023/6/6 下午5:23
@summary: 使用text cnn 进行角色分类
"""
import os, sys
import time

import pandas as pd
import torch
import numpy as np
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
from tqdm import tqdm
from sklearn import metrics
from datetime import timedelta
from transformers import BertTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataloader import TextDatasetMulti
from utils.dataloader import BatchTextDataset
from models.role_extract.text_cnn import TextCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = ''

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
    dataset_batch = BatchTextDataset(tokenizer, max_len=256)

    train_dataset = TextDatasetMulti(f"{data_path}/data/role/train_data.csv")

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)
    dev_dataset = TextDatasetMulti(f"{data_path}/data/role/dev_data.csv")

    dev_dataloader = DataLoader(dev_dataset, batch_size=128, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)
    test_dataset = TextDatasetMulti(f"{data_path}/data/role/test_data.csv")

    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)
    return train_dataloader, dev_dataloader, test_dataloader

# 模型评估
def evaluation(model, test_dataloader, loss_func, device):
    """"""
    model.eval()
    total_loss = 0
    predict_all = []
    labels_all = []
    for ind, (token, label) in enumerate(test_dataloader):
        token = token.to(device)
        label = label.to(device)
        out = model(token)
        label = label.float()
        loss = loss_func(out, label)
        total_loss += loss.detach().item()
        label = label.data.cpu().numpy()
        # pred_label = out.data.numpy()
        pred_label = out.data.cpu().sigmoid().numpy()
        for l in label:
            labels_all.extend(l.tolist())
        for pl in pred_label:
            pl_ = [1 if p > 0.6 else 0 for p in pl.tolist()]
            predict_all.extend(pl_)
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, total_loss / len(test_dataloader)


def train(load_model=False):
    start_time = time.time()
    tokenizer = load_tokenizer(f"{data_path}/lib/pretrained/albert_chinese_base")
    train_dataloader, valid_dataloader, test_dataloader = get_train_dataloader(tokenizer)
    label_len = 11
    vocab_size, num_classes, num_kernels, kernel_size = 21128, 11, 100, [3, 4, 5]
    model = TextCNN(vocab_size, num_classes, num_kernels, kernel_size)

    model_dir = f"../model_file/role"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if load_model:
        model.load_state_dict(torch.load(f"{model_dir}/text_cnn_best.pt", map_location=torch.device('cpu')))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.00005,
                                  betas=(0.9, 0.999),
                                  eps=1e-08,
                                  weight_decay=0.01,
                                  amsgrad=False)
    loss_func = F.cross_entropy
    # loss_func = F.binary_cross_entropy_with_logits
    loss_total, top_acc = [], 0
    valid_best_loss = float('inf')
    last_improve = 0
    total_batch = 0
    stop_flag = False
    train_acc_history, train_loss_history = [], []
    valid_acc_history, valid_loss_history = [], []
    epochs = 10
    for epoch in range(epochs):
        print("epoch [{}/{}]".format(epoch + 1, epochs))
        model.train()
        epoch_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        for i, (token, label) in enumerate(tqdm_bar):
            # data to device
            # print(label)
            token = token.to(device)
            label = label.to(device)
            model.zero_grad()
            out = model(token)

            label = label.float()
            loss = loss_func(out, label)
            loss.backward()
            optimizer.step()
            loss_total.append(loss.detach().item())
            # valid_acc, valid_loss = evaluation(tcml, valid_dataloader, loss_func, device)
            # print(valid_acc, valid_loss)
            # break
            if total_batch % 1000 == 0 and total_batch != 0:
                valid_acc, valid_loss = evaluation(model, valid_dataloader, loss_func, device)
                true_label = label.data.cpu().numpy()
                # pred_label = out.data.numpy()
                pred_label = out.data.cpu().sigmoid().numpy()
                true_label_, pred_label_ = [], []
                for tl in true_label:
                    true_label_.extend(tl.tolist())
                for pl in pred_label:
                    pl_ = [1 if p > 0.6 else 0 for p in pl.tolist()]
                    pred_label_.extend(pl_)

                train_acc = metrics.accuracy_score(true_label_, pred_label_)

                train_acc_history.append(train_acc)
                train_loss_history.append(loss.detach().item())
                valid_acc_history.append(valid_acc)
                valid_loss_history.append(valid_loss)

                if epoch and epoch % 5 == 0:
                    save_model_path = f"{model_dir}/text_cnn_{epoch}.pt"
                    torch.save(model.state_dict(), save_model_path)

                # evaluate on validate data
                if valid_loss < valid_best_loss:
                    valid_best_loss = valid_loss
                    # 保存最好好的模型
                    best_model_path = f"{model_dir}/text_cnn_best.pt"
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
            if total_batch - last_improve > 2000:
                print('No optimization for a long time, auto-stopping......')
                stop_flag = True
                break
        # 最后一轮保存模型
        if epoch == epochs - 1:
            best_model_path = f"{model_dir}/text_cnn_best.pt"
            torch.save(model.state_dict(), best_model_path)
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))
        if stop_flag:
            break
        time.sleep(0.5)

def predict(text, name):
    """"""
    tokenizer = load_tokenizer(f"{data_path}/lib/pretrained/albert_chinese_base")
    label_len = 11
    vocab_size, num_classes, num_kernels, kernel_size = 21128, 11, 100, [3, 4, 5]
    max_len, split_len = 256, 3
    model = TextCNN(vocab_size, num_classes, num_kernels, kernel_size)
    best_model_path = f"../model_file/role/text_cnn_best.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        text_split = text.split(name)
        text_split = [t for t in text_split if t]
        token_list = []
        for j, ts in enumerate(text_split):
            if not ts:
                continue
            text_dict = tokenizer(ts, truncation=True, max_length=max_len, padding='max_length')
            token_list.append(text_dict['input_ids'])
        if len(token_list) >= split_len:
            token_list = token_list[:split_len]
        else:
            for k in range(split_len - len(token_list)):
                token_list.append([0] * max_len)

        input_ids = torch.tensor([token_list])

        out = model(input_ids)
        pred_label = out.data.cpu().sigmoid().numpy()
        # print(pred_label)
        pl = [1 if p > 0.95 else 0 for p in pred_label[0].tolist()]
        print(pl)
        return pl


if __name__ == '__main__':
    """"""
    # train()
    # role = ["minjing", "baoan", "souhai", "xianyi", "faxian", "zhengren", "toushu", "beitoushu",
    #         "jubao", "beijubao", "qita"]
    # text = ""
    # name = ""
    # res = predict(text, name)

