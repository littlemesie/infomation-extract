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
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import DataLoader
from utils.dataloader import DatasetNER
from utils.dataloader import BertNERBatchDataset
from utils.time_util import get_time_diff
from utils.load_util import load_tokenizer
from models.ner_extract.bert_bilstm_crf import BertBilstmCRF

tag2ids = {
    'O': 0,
    'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3,
    'B-RACE': 4, 'M-RACE': 5, 'E-RACE': 6,
    'B-ORG': 7, 'M-ORG': 8, 'E-ORG': 9,
    'B-TITLE': 10, 'M-TITLE': 11, 'E-TITLE': 12,
    'B-CONT': 13, 'M-CONT': 14, 'E-CONT': 15,
    'B-PRO': 16, 'M-PRO': 17, 'E-PRO': 18,
    'B-EDU': 19, 'M-EDU': 20, 'E-EDU': 21,
    'B-LOC': 22, 'M-LOC': 23, 'E-LOC': 24,
    'S-NAME': 25, 'S-RACE': 26, 'S-ORG': 27
}
ids2tags = {v: k for k, v in tag2ids.items()}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrained_model_path = "/home/mesie/python/aia-nlp-service/lib/pretrained/albert_chinese_base"
# pretrained_model_path = "/home/yons/python/conf/lib/pretrained/albert_chinese_base"

def get_train_dataloader(tokenizer):
    dataset_batch = BertNERBatchDataset(tokenizer, tag2ids, max_seq_len=512)

    train_dataset = DatasetNER(f"../data/ner/train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    valid_dataset = DatasetNER(f"../data/ner/dev.csv")
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True,
                                  num_workers=2, collate_fn=dataset_batch)

    return train_dataloader, valid_dataloader

def evaluation(model, valid_dataloader):
    model.eval()
    preds = []
    trues = []
    total_loss = 0
    for step, batch_data in enumerate(tqdm(valid_dataloader)):
        token_ids = batch_data['token_ids'].to(device)
        token_type_ids = batch_data['token_type_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        labels = batch_data['tags'].to(device)
        output = model(token_ids, attention_mask, token_type_ids, labels=labels)
        logits = output.logits
        total_loss += output.loss.detach().item()
        attention_mask = attention_mask.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        batch_size = token_ids.size(0)
        for i in range(batch_size):
            length = sum(attention_mask[i])
            logit = logits[i][1:length]
            logit = [ids2tags[i] for i in logit]
            label = labels[i][1:length]
            label = [ids2tags[i] for i in label]
            preds.append(logit)
            trues.append(label)

    report = classification_report(trues, preds)
    return report, total_loss / len(valid_dataloader)

def train(tokenizer, load_model=False):
    start_time = time.time()
    train_dataloader, valid_dataloader = get_train_dataloader(tokenizer)
    num_labels, max_seq_len = len(tag2ids), 512
    model = BertBilstmCRF(num_labels, max_seq_len, pretrained_model_path, device)

    model_dir = f"../model_file/ner"
    best_model_path = f"{model_dir}/bert_bilstm_crf_best.pt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if load_model:
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.to(device)
    # parameters
    no_decay = ["bias", "LayerNorm.weight"]
    bert_learning_rate = 3e-5
    crf_learning_rate = 3e-3
    adam_epsilon = 1e-8
    weight_decay = 0.01
    model_param = list(model.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []
    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': bert_learning_rate},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': crf_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': crf_learning_rate},
    ]
    optimizer = torch.optim.Adam(optimizer_parameters, lr=bert_learning_rate, eps=adam_epsilon)
    total_batch = 0
    epochs = 1
    valid_best_loss = 0.0001
    last_improve = 0
    model.train()
    for epoch in range(epochs):
        print("epoch [{}/{}]".format(epoch + 1, epochs))
        loss_total = []
        epoch_time = time.time()
        tqdm_bar = tqdm(train_dataloader, desc="training epoch{epoch}".format(epoch=epoch))
        for i, batch_data in enumerate(tqdm_bar):
            token_ids = batch_data['token_ids'].to(device)
            token_type_ids = batch_data['token_type_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            tags = batch_data['tags'].to(device)
            output = model(token_ids, attention_mask, token_type_ids, labels=tags)
            loss = output.loss
            loss_total.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if total_batch % 500 == 0 and total_batch != 0:
                report, valid_loss = evaluation(model, valid_dataloader)
                # if epoch and epoch % 5 == 0:
                #     save_model_path = f"{model_dir}/bert_bilstm_crf_best_{epoch}.pt.pt"
                    # torch.save(model.state_dict(), save_model_path)

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
                msg = "Iter：{0:6}, Train_Loss: {1:>5.2}, Val_Loss: {2:5.2}, Time: {3} {4}"
                print(msg.format(total_batch, loss.item(), valid_loss, time_diff, improve))
                model.train()
            total_batch += 1
            # if total_batch - last_improve > 500:
            #     print('No optimization for a long time, auto-stopping......')
            #     stop_flag = True
            #     break
            # break
        # 最后一轮保存模型
        if epoch == epochs - 1:
            torch.save(model.state_dict(), best_model_path)
        print("Epoch: %03d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss_total), time.time() - start_time))

def ner_tokenizer(text, tokenizer, max_len=512):
    text = text[:max_len - 2]
    text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
    encoded = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=True)
    token_ids = torch.tensor(encoded['input_ids']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(encoded['token_type_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0).to(device)
    return token_ids, token_type_ids, attention_mask

def predict(tokenizer, text):
    num_labels, max_seq_len = len(tag2ids), 512
    model = BertBilstmCRF(num_labels, max_seq_len, pretrained_model_path, device)
    best_model_path = f"../model_file/ner/bert_bilstm_crf_best.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    import time
    t1 = time.time()
    token_ids, token_type_ids, attention_mask = ner_tokenizer(text, tokenizer, max_seq_len)

    output = model(token_ids, attention_mask, token_type_ids)
    attention_mask = attention_mask.detach().cpu().numpy()
    length = sum(attention_mask[0])
    logits = output.logits
    logits = logits[0][1:length - 1]
    logits = [ids2tags[i] for i in logits]
    print(logits)
    entities = get_entities(logits)
    print(entities)
    print(time.time() - t1)

if __name__ == '__main__':
    """"""
    tokenizer = load_tokenizer(pretrained_model_path)
    # train(tokenizer)
    text = "孙悟空，男，汉族，出生与10万年前"
    predict(tokenizer, text)
