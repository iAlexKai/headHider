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

    # def forward(self, context, epsilon):
    def forward(self, context):

        batch_size, _ = context.size()  # prior: (batch, 4 * hidden)
        # return context, context, context

        # import pdb
        # pdb.set_trace()
        context = self.fc(context)

        # return context, context, context


        mu = self.context_to_mu(context)
        logsigma = self.context_to_logsigma(context)


        std = torch.exp(0.5 * logsigma)


        epsilon = to_tensor(torch.ones([batch_size, self.z_size]))
        # return mu, mu, logsigma

        z = epsilon * std + mu
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
    def forward(self, init_hidden, decoder_input):   # 所有的Tensor必须用到，同时converter转换正确
        # batch_size = init_hidden.size(0)
        # decoder_input = to_tensor(torch.LongTensor([[go_id]]).view(1, 1))  # (batch, 1)
        # inputs = self.embedding(inputs)
        #
        # if context is not None:
        #     repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1)
        #     inputs = torch.cat([inputs, repeated_context], 2)
        #
        # # inputs = F.dropout(inputs, 0.5, self.training)
        #
        # hids, h_n = self.rnn(inputs, init_hidden.unsqueeze(0))
        # decoded = self.out(hids.contiguous().view(-1, self.hidden_size))  # reshape before linear over vocab
        # decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        # return decoded

        # batch_size = init_hidden.size(0)
        ##################################################################
        # Alex
        # torch 1.8.0+ has supported embedding with input type IntTensor
        # Tensorrt only support int32.
        ##################################################################
        # decoder_input = to_tensor(torch.LongTensor([[go_id]]).view(1, 1))  # (batch, 1)
        # batch_size = decoder_input.shape[0]

        decoder_input = self.embedding(decoder_input)  # (batch, 1, emb_dim)
        decoder_hidden = init_hidden.view(1, 1, 1600)  # (1, batch, 4*hidden+z_size)
                                                       # 这里，不能写成 (1, 1, -1) view的converter里面-1没有处理好

        # pred_outs = torch.ones([10]).cuda()

        # for di in range(10):
        decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)  # (1, 1, hidden)

        decoder_output = self.out(decoder_output.contiguous().view(1, self.hidden_size))  # (1, vocab_size)
        topi = decoder_output.max(dim=1, keepdim=True)[1]
        return topi
        # import pdb
        # pdb.set_trace()
        pred_outs[di] = topi[0][0]
        decoder_input = self.embedding(topi)

        # import pdb
        # pdb.set_trace()
        return pred_outs


    # 生成结果，可以直接当做test的输出，也可以做evaluate的metric值（如BLEU）计算
    # init_hidden是prior_z cat c
    # batch_size是1，一次只测试一首诗

    # init_hidden (prior_z和c的cat)
    # max_len： config.maxlen 即10
    # SOS_tok: 即<s>对应的token
    def testing(self, init_hidden, maxlen, go_id, mode="greedy"):
        batch_size = init_hidden.size(0)
        assert batch_size == 1

        decoder_input = to_tensor(torch.LongTensor([[go_id]]).view(1, 1))  # (batch, 1)
        # import pdb
        # pdb.set_trace()

        # input: (batch=1, len=1, emb_size)
        decoder_input = self.embedding(decoder_input)  # (batch, 1, emb_dim)
        # hidden: (batch=1, 2, hidden_size * 2)
        decoder_hidden = init_hidden.unsqueeze(0)  # (1, batch, 4*hidden+z_size)
        # pred_outs = np.zeros((batch_size, maxlen), dtype=np.int64)
        pred_outs = []
        for di in range(maxlen - 1):  # decode要的是从<s>后一位开始，因此总长度是max_len-1
            # 输入decoder
            decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)  # (1, 1, hidden)

            decoder_output = self.out(decoder_output.contiguous().view(-1, self.hidden_size))  # (1, vocab_size)
            topi = decoder_output.max(1, keepdim=True)[1]
            # 拿到pred_outs以返回

            # 为下一次decode准备输入字
            # if di != 0:
            decoder_input = self.embedding(topi)

            # ni = topi.squeeze().cpu().numpy()
            # pred_outs[:, di] = ni

            pred_outs.append(topi)
            #     pred_outs[:, di] = header.item()
        # 结束for完成一句诗的token预测
        return pred_outs
