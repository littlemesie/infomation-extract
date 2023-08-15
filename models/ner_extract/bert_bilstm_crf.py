# -*- coding:utf-8 -*-

"""
@date: 2023/8/3 下午4:47
@summary:
"""
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig

class ModelOutput:
  def __init__(self, logits, labels, loss=None):
    self.logits = logits
    self.labels = labels
    self.loss = loss

class BertBilstmCRF(nn.Module):
  def __init__(self, num_labels, max_seq_len, pretrained_model_path, device, lstm_hidden=128, num_layers=1,
               dropout=0.1):
    super(BertBilstmCRF, self).__init__()
    self.bert = BertModel.from_pretrained(pretrained_model_path)
    self.bert_config = BertConfig.from_pretrained(pretrained_model_path)
    hidden_size = self.bert_config.hidden_size
    self.lstm_hidden = lstm_hidden
    self.device = device
    self.max_seq_len = max_seq_len
    self.num_layers = num_layers
    self.bilstm = nn.LSTM(hidden_size, self.lstm_hidden, self.num_layers, bidirectional=True, batch_first=True,
               dropout=dropout)
    self.linear = nn.Linear(self.lstm_hidden * 2, num_labels)
    self.crf = CRF(num_labels, batch_first=True)

  def init_hidden(self, batch_size):
    h0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
    c0 = torch.randn(2 * self.num_layers, batch_size, self.lstm_hidden, requires_grad=True).to(self.device)
    return h0, c0

  def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    seq_out = bert_output[0]  # [batch_size, max_seq_len, 768]
    batch_size = seq_out.size(0)
    h0, c0 = self.init_hidden(batch_size)
    seq_out, _ = self.bilstm(seq_out, (h0, c0))
    seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)  # [batch_size, max_seq_len, num_tags]
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    out = ModelOutput(logits, labels, loss)
    return out