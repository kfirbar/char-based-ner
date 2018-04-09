import torch
import numpy as np
import os
from torch import nn, autograd, optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import random
import json
from collections import Counter
import pickle
import pandas as pd
import re
import codecs
import argparse
import sys
import timeit


# Vocab class, containing char ids and tag ids
class Lang:
    def __init__(self):
        self.char2id = {"__pad__": 0}
        self.id2char = {0: "__pad__"}
        self.char2count = {"__pad__": 0}
        self.n_chars = 1
        self.tag2id = {"O": 0}
        self.id2tag = {0: "O"}
        self.tag2count = {"O": 0}
        self.n_tags = 1

    def get_tag_id(self, tag):
        if tag in self.tag2id:
            return self.tag2id[tag]
        self.tag2id[tag] = self.n_tags
        self.tag2count[tag] = 1
        self.id2tag[self.n_tags] = tag
        self.n_tags += 1
        return self.tag2id[tag]

    def get_char_id(self, c):
        if c in self.char2id:
            self.char2count[c] += 1
            return self.char2id[c]
        self.char2id[c] = self.n_chars
        self.char2count[c] = 1
        self.id2char[self.n_chars] = c
        self.n_chars += 1
        return self.char2id[c]


# the Model RNN class
class CharBasedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, output_size, n_layers, use_gpu = False):
        super(CharBasedRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.use_gpu = use_gpu

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, bidirectional=True)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_batch):
        if self.use_gpu:
            seq_lengths = torch.LongTensor(map(len, input_batch)).cuda()
            seq_tensor = Variable(torch.zeros((len(input_batch), seq_lengths.max())).cuda()).long()
        else:
            seq_lengths = torch.LongTensor(map(len, input_batch))
            seq_tensor = Variable(torch.zeros((len(input_batch), seq_lengths.max()))).long()

        for idx, (seq, seqlen) in enumerate(zip(input_batch, seq_lengths)):
            if self.use_gpu:
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq).cuda()
            else:
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)


        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        seq_tensor = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        if self.use_gpu:
            seq_tensor = seq_tensor.cuda()

        seq_tensor = self.embedding(seq_tensor)

        packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())

        packed_output, (ht, ct) = self.lstm(packed_input)

        output, _ = pad_packed_sequence(packed_output)

        output = output.transpose(0, 1)
        max_seq = output.data.shape[0]
        batch_size = output.data.shape[1]

        final_outputs = []
        for o in xrange(0, output.data.shape[0]):
            final_outputs.append(self.out(output[o]))
        return final_outputs, perm_idx

    def init_hidden(self):
        if self.use_gpu:
            return (Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda()),
                    Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size).cuda()))
        else:
            return (Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers * 2, 1, self.hidden_size)))
