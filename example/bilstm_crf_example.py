# -*- coding:utf-8 -*-

"""
@date: 2023/10/19 上午9:27
@summary: 基于bilstm crf的实体提取
"""
import os
import sys
import time
import torch
import pandas as pd
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import torch.optim as optim
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities
from models.ner_extract.bilstm_crf import BilstmCRF

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

def extend_maps(tag2ids):
    tag2ids['[PAD]'] = len(tag2ids) if not tag2ids.get('[PAD]', None) else tag2ids['[PAD]']
    tag2ids['[UNK]'] = len(tag2ids) if not tag2ids.get('[UNK]', None) else tag2ids['[UNK]']
    return tag2ids

tag2ids = extend_maps(tag2ids)

def get_word2id(vocab_path):
    """"""
    w2i = {}
    with open(vocab_path, mode="r", encoding="utf-8") as reader:
        for index, line in enumerate(reader):
            w = line.strip("\r\n").split()[0] if line.strip() else line.strip("\r\n")
            w2i[w] = index

    return w2i

def build_corpus(data_path):
    """读取数据csv"""
    word_lists = []
    tag_lists = []
    data_df = pd.read_csv(data_path)
    for _, row in data_df.iterrows():
        word_list = list(row['text'])
        tag_list = [t for t in row['tags'].split(' ')]
        if len(tag_list) < len(word_list):
            tag_list.extend(['O']*(len(word_list) - len(tag_list)))
        elif len(tag_list) > len(word_list):
            tag_list = tag_list[:len(word_list)]
        word_lists.append(word_list)
        tag_lists.append(tag_list)
    return word_lists, tag_lists

def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices

def tensorized(batch, maps):
    PAD = maps.get('[PAD]')
    UNK = maps.get('[UNK]')
    max_len = len(batch[0])
    batch_size = len(batch)
    # print(PAD, UNK, max_len, batch_size)
    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths

def evaluation(model, batch_size, dev_word_lists, dev_tag_lists, words2ids, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        val_losses = 0.
        val_step = 0
        for ind in range(0, len(dev_word_lists), batch_size):
            val_step += 1
            batch_texts = dev_word_lists[ind:ind + batch_size]
            batch_tags = dev_tag_lists[ind:ind + batch_size]
            # 准备batch数据
            batch_texts, lengths = tensorized(batch_texts, words2ids)
            batch_texts = batch_texts.to(device)
            batch_tags, lengths = tensorized(batch_tags, tag2ids)
            batch_tags = batch_tags.to(device)

            # forward
            output = model(batch_texts, batch_tags)
            val_losses += output.loss.item()

            logits = output.logits
            labels = batch_tags.detach().cpu().numpy()
            batch_size = batch_texts.size(0)
            for i in range(batch_size):
                logit = [ids2tags[j] for j in logits[i]]
                label = [ids2tags[j] for j in labels[i]]
                preds.append(logit)
                trues.append(label)
        val_loss = val_losses / val_step
        report = classification_report(trues, preds)
        return val_loss, report


def train(load_model=False):

    words2ids = get_word2id(f"{pretrained_model_path}/vocab.txt")
    train_word_lists, train_tag_lists = build_corpus(f"../data/ner/train.csv")
    dev_word_lists, dev_tag_lists = build_corpus(f"../data/ner/dev.csv")
    # sort text
    train_word_lists, train_tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
    # print(train_word_lists)
    dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)
    # 参数
    vocab_size, num_labels = len(words2ids), len(tag2ids)
    emb_size, hidden_size, num_layers, dropout = 128, 128, 1, 0.5
    batch_size, print_step = 64, 20
    # model
    model = BilstmCRF(vocab_size, emb_size, hidden_size, num_labels, num_layers=num_layers, dropout=dropout)
    # model path
    model_dir = f"../model_file/ner"
    best_model_path = f"{model_dir}/bilstm_crf_best.pt"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if load_model:
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.to(device)
    # train
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 30
    best_val_loss = 0.001
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        step = 0
        losses = 0.
        model.train()
        for ind in range(0, len(train_word_lists), batch_size):
            batch_texts = train_word_lists[ind:ind + batch_size]
            batch_tags = train_tag_lists[ind:ind + batch_size]
            # 准备数据
            batch_texts, lengths = tensorized(batch_texts, words2ids)
            batch_texts = batch_texts.to(device)
            batch_tags, lengths = tensorized(batch_tags, tag2ids)
            batch_tags = batch_tags.to(device)
            # forward
            output = model(batch_texts, batch_tags)
            loss = output.loss
            losses += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % print_step == 0:
                total_step = (len(train_word_lists) // batch_size + 1)
                print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(epoch, step, total_step,
                                                                                    100. * step / total_step,
                                                                                    losses / print_step))
                losses = 0.
        # 每轮结束测试在验证集上的性能
        val_loss, report = evaluation(model, batch_size, dev_word_lists, dev_tag_lists, words2ids, device)
        print("Epoch {}, Val Loss:{:.4f}".format(epoch, val_loss))
        print(report)

        if val_loss < best_val_loss:
            print("保存模型...")
            torch.save(model.state_dict(), best_model_path)
            break
        # 最后一轮保存模型
        if epoch == epochs:
            torch.save(model.state_dict(), best_model_path)

    print("训练完毕,共用时{}秒.".format(int(time.time() - start_time)))

def predict(text):
    words2ids = get_word2id(f"{pretrained_model_path}/vocab.txt")
    vocab_size, num_labels = len(words2ids), len(tag2ids)
    emb_size, hidden_size, num_layers, dropout = 128, 128, 1, 0.5
    model = BilstmCRF(vocab_size, emb_size, hidden_size, num_labels, num_layers=num_layers, dropout=dropout)
    best_model_path = f"../model_file/ner/bilstm_crf_best.pt"
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    model.eval()
    import time
    t1 = time.time()
    word_list = list(text)
    tensorized_text, lengths = tensorized([word_list], words2ids)
    tensorized_sents = tensorized_text.to(device)
    with torch.no_grad():
        output = model(tensorized_sents)
    logits = output.logits
    logits = logits[0]
    fc_out = output.fc_out.squeeze(0)
    prob_list = []
    for i, fo in enumerate(fc_out):
        prob_list.append(list(fo.sigmoid().cpu().numpy())[logits[i]])
        # print(list(fo.sigmoid().cpu().numpy()))
    print(logits)
    print(prob_list)
    logits = [ids2tags[i] for i in logits]
    print(logits)
    entities = get_entities(logits)
    sigmoid_threshold = 0.98
    entities_ = entities.copy()
    for entity in entities:
        prob = 0
        j = 0
        for i in range(entity[1], entity[2]):
            prob += prob_list[i]
            j += 1
        prob = prob / j if j != 0 else 0
        print(prob)
        if prob < sigmoid_threshold:
            entities_.remove(entity)
    print(entities_)
    print(time.time() - t1)

if __name__ == '__main__':
    """"""
    # train(load_model=True)
    text = "孙悟空，男，汉族，出生与10万年前"
    # text = "孙悟空和杨戬谁更厉害"
    predict(text)