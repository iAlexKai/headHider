"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import to_tensor


# 编码topic和上一句
class Encoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidirectional, n_layers, noise_radius=0.2):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.noise_radius = noise_radius
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, inputs):

        # if self.embedding is not None:
        inputs = self.embedding(inputs)  # 过embedding
        
        batch_size, seq_len, emb_size = inputs.size()  # (batch, len, emb_size) len是12，即标题的最大长度
        # import pdb
        # pdb.set_trace()
        hids, h_n = self.rnn(inputs)

        # enc = h_n.transpose(1, 0).contiguous().view(batch_size, -1)
        enc = h_n.view(batch_size, -1)

        return enc


class Variation(nn.Module):
    def __init__(self, input_size, z_size, dropout_rate, init_weight):
        super(Variation, self).__init__()
        self.input_size = input_size
        self.z_size=z_size
        self.init_w = init_weight
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1200),
            nn.BatchNorm1d(1200, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(1200, z_size),
            nn.BatchNorm1d(z_size, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.1),
            # nn.Dropout(dropout_rate),
        )
        self.context_to_mu = nn.Linear(z_size, z_size)  # activation???
        self.context_to_logsigma = nn.Linear(z_size, z_size)
        
        self.fc.apply(self.init_weights)
        self.init_weights(self.context_to_mu)
        self.init_weights(self.context_to_logsigma)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-self.init_w, self.init_w)
            m.bias.data.fill_(0)

    def forward(self, context, epsilon):
    # def forward(self, context):

        batch_size, _ = context.size()  # prior: (batch, 4 * hidden)
        # return context, context, context

        # import pdb
        # pdb.set_trace()
        context = self.fc(context)

        mu = self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context)

        std = torch.exp(0.5 * logsigma)

        # epsilon = to_tensor(torch.ones([batch_size, self.z_size]))
        # import pdb
        # pdb.set_trace()
        # rand = np.random.randn(1, self.z_size)
        # print(rand)
        # for i in range(self.z_size):
        #     epsilon[0][i] = rand[0][i]
        # return mu, mu, logsigma

        z = epsilon * std
        z = z + mu
        print("Done processing the model.")
        return z, mu, logsigma
    

class Decoder(nn.Module):
    # Decoder(self.embedder, config.emb_size, config.n_hidden*4 + config.z_size, self.vocab_size, n_layers=1)
    def __init__(self, embedder, input_size, hidden_size, vocab_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_size= input_size 
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size

        self.embedding = embedder
        # 给decoder的init_hidden加一层非线性变换relu
        # encoder可以是双向的GRU，但decoder一定是单向的
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    # init_hidden: (batch, z_size + 4*hidden) --unsqueeze->  (1, batch, z_size+4*hidden)
    # self.decoder(torch.cat((z, c), 1), None, target[:, :-1], target_lens-1)
    def forward(self, decoder_input, init_hidden):   # 所有的Tensor必须用到，同时converter转换正确
        ##################################################################
        # Alex
        # torch 1.8.0+ has supported embedding with input type IntTensor
        # Tensorrt only support int32.
        ##################################################################

        decoder_input = self.embedding(decoder_input)  # (batch, 1, emb_dim)
        decoder_output, decoder_hidden = self.rnn(decoder_input, init_hidden)  # (1, 1, hidden)

        decoder_output = self.out(decoder_output.contiguous().view(1, self.hidden_size))  # (1, vocab_size)
        topi = decoder_output.max(dim=1, keepdim=True)[1]
        
        return topi, decoder_hidden  # 用于下一轮的输入
        
        
        # import pdb
        # pdb.set_trace()
        pred_outs[di] = topi[0][0]
        decoder_input = self.embedding(topi)

        # import pdb
        # pdb.set_trace()
        return pred_outs
