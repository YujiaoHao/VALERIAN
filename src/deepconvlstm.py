# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:05:58 2022
remove last output layer, keep feature extractor only
@author: haoyu
"""

import torch
from torch import nn
import torch.nn.functional as F

NUM_CHANNEL = 6
WINDOW_LENGTH = 200
train_on_gpu = True


class HARModel(nn.Module):

    def __init__(self, n_hidden=128, n_layers=1, n_filters=64,
                 filter_size=5, drop_prob=0.5, pretrained=False):
        super(HARModel, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters

        self.filter_size = filter_size

        self.conv1 = nn.Conv1d(NUM_CHANNEL, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size, padding=int(filter_size / 2))

        self.lstm1 = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden, batch_size=32):

        x = x.view(-1, NUM_CHANNEL, WINDOW_LENGTH)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(batch_size, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)

        x = x.contiguous().view(-1, self.n_hidden)
        out = self.dropout(x)
        
        # out = x.view(batch_size, -1, self.n_classes)[:, -1, :]

        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden