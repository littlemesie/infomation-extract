# -*- coding:utf-8 -*-

"""
@date: 2023/7/31 下午6:07
@summary:
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
from utils.dataloader import DatasetRelation
from utils.dataloader import TextRelationBatchDataset
from utils.time_util import get_time_diff
from utils.load_util import load_tokenizer
from models.relation_extract.text_cnn_classfication import TextCNNRE

"""
一男（
）称其妹妹联系不上，担心出事，其妹妹名叫：
 ，在郫县指挥中心上班。
 刘刚 
"""
relation_list = ['unknown', '祖孙', '同门', '上下级', '师生', '情侣', '亲戚', '夫妻', '好友', '兄弟姐妹', '父母', '合作']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model_path = "/home/mesie/python/aia-nlp-service/lib/pretrained/albert_chinese_base"


def get_train_dataloader(tokenizer):
    dataset_batch = TextRelationBatchDataset(tokenizer, max_len=256)

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
    for ind, (e_token, labels) in enumerate(valid_dataloader):
        e_token = e_token.to(device)
        labels = labels.to(device)

        out = model(e_token)
        loss = criterion(out, labels)
        total_loss += loss.detach().item()

        label = labels.data.cpu().numpy()
        pred = torch.max(out.data.cpu(), 1)[1].numpy()
        label_list = np.append(label_list, label)
        pred_list = np.append(pred_list, pred)

    acc = metrics.accuracy_score(label_list, pred_list)

    return acc, total_loss / len(valid_dataloader)


def train(tokenizer, load_model=False):
    start_time = time.time()
    train_dataloader, valid_dataloader = get_train_dataloader(tokenizer)
    vocab_size, num_classes, num_kernels, kernel_size = 21128, len(relation_list), 100, [3, 4, 5]
    model = TextCNNRE(vocab_size, num_classes, num_kernels, kernel_size)

    model_dir = f"../model_file/relation"
    best_model_path = f"{model_dir}/text_cnn_relation_best.pt"
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
    epochs = 10
    for epoch in range(epochs):
        print("epoch [{}/{}]".format(epoch + 1, epochs))
        model.train()
        epoch_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        for i, (e_token, labels) in enumerate(tqdm_bar):
            e_token = e_token.to(device)
            labels = labels.to(device)
            model.zero_grad()
            out = model(e_token)
            # print(out.shape, labels.shape)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0 and total_batch != 0:
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

if __name__ == '__main__':
    """"""

    tokenizer = load_tokenizer(pretrained_model_path)
    train(tokenizer)



