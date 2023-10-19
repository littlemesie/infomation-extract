# -*- coding:utf-8 -*-

"""
@date: 2023/10/18 下午5:43
@summary:
"""
class ModelOutput:
  def __init__(self, logits, labels, loss=None, fc_out=None):
    self.logits = logits
    self.labels = labels
    self.loss = loss
    self.fc_out = fc_out
