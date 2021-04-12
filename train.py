import argparse
import time
from datetime import datetime
import numpy as np
import random
import json
import torch
import pickle
import os, sys
import time
import pdb
pdb.set_trace()
from torch2trt_dynamic import torch2trt_dynamic

from configs import Config as Config
from models.poemwae import PoemWAE

from helper import to_tensor, timeSince  # 将numpy转为tensor


def get_one_input_for_tensorrt(rev_vocab, title_size):
    title = "春暖花开"
    title = [rev_vocab.get(item, rev_vocab["<unk>"]) for item in title]
    title_batch = [title + [0] * (title_size - len(title))]
    headers_batch = []
    for i in range(4):
        headers_batch.append(title[i])
    title_tensor = to_tensor((np.array(title_batch, dtype=np.int32)))
    # import pdb
    # pdb.set_trace()
    # headers_batch = to_tensor((np.array(headers_batch)))
    # title = to_tensor((np.array(title)))

    # return title_tensor, headers_batch, title
    return title_tensor


def get_user_input(rev_vocab, title_size, last_file_length):
    import time
    def _is_Chinese(title):
        for ch in title:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False
    while True:
        cur_file = open('/home/nano/Desktop/workspace_myk/shared_input_zone.txt').read().strip().split('\n')
        if len(cur_file) == last_file_length:
            #print("file not changed, keep waiting for another 3 seconds")
            time.sleep(1)
            continue
        last_file_length = len(cur_file)
        title = cur_file[-1]
       
        if title is None or title is "" or len(title) != 4 or not _is_Chinese(title):
            print("The title must be four-word.")
            continue
        else:
            break

    title = [rev_vocab.get(item, rev_vocab["<unk>"]) for item in title]
    title_batch = [title + [0] * (title_size - len(title))]

    headers_batch = []
    for i in range(4):
        headers_batch.append([[title[i]]])

    return (np.array(title_batch, dtype=np.int32), headers_batch, title), last_file_length


def main():

    # config for training
    config = Config()
    vocab_file = open('pickle_file/vocab.pickle', 'rb')
    rev_vocab_file = open('pickle_file/rev_vocab.pickle', 'rb')
    vocab = pickle.load(vocab_file)
    rev_vocab = pickle.load(rev_vocab_file)

    model = PoemWAE(config=config, vocab=vocab, rev_vocab=rev_vocab)

    import time
    time_start = time.time()
    print("\nbefore loading model\n")

    model.load_state_dict(torch.load(f='./output/basic/model_state_dict.pckl'))
    model = model.cuda()
    print("finish loading model, using {:d} seconds".format(int(time.time()-time_start)))

    model.vocab = vocab
    model.rev_vocab = rev_vocab

    # 从test_loader里面拿到一个输入，使用tensorrt对模型进行压缩
    title_tensor = get_one_input_for_tensorrt(rev_vocab, config.title_size)
    decoder_input = to_tensor(torch.IntTensor([[rev_vocab['<s>']]]).view(1, 1))  # (batch, 1)
    decoder_init = torch.zeros([1, 1, 1600]).cuda()
    # epsilon = torch.randn([1, config.z_size]).cuda()

    # model = torch2trt_dynamic(model, [title_tensor, epsilon, decoder_input, decoder_init], max_workspace_size=1 << 28)  # max_workspace_size 最大不能超过30
    model = torch2trt_dynamic(model, [title_tensor, decoder_input, decoder_init],
                              max_workspace_size=1 << 28)  # max_workspace_size 最大不能超过30
    print("Finish model_trt build")
    last_title = None
    last_file_length = 0
    while True:
        model.eval()  # eval()主要影响BatchNorm, dropout等操作
        print("Waiting for vocal input")
        batch, last_file_length = get_user_input(rev_vocab, config.title_size, last_file_length)

        if batch is None:
            break

        title_list, headers, title = batch  # batch size是1，一个batch写一首诗

        if title == last_title:
            continue
        last_title = title

        title_tensor = to_tensor(title_list)

        # test函数将当前batch对应的这首诗decode出来，记住每次decode的输入context是上一次的结果
        # import pdb
        # pdb.set_trace()
        print("before inferencing")
        import time
        time_start = time.time()

        decoder_input = to_tensor(torch.IntTensor([[rev_vocab['<s>']]]).view(1, 1))  # (batch, 1)
        decoder_init = torch.zeros([1, 1, 1600]).cuda()
        # epsilon = torch.randn([1, config.z_size]).cuda()
        word_list = []
        for _ in range(10):
            # topi, decoder_hidden = model(title_tensor, epsilon, decoder_input, decoder_init)
            topi, decoder_hidden = model(title_tensor, decoder_input, decoder_init)
            # import pdb
            # pdb.set_trace()
            word_list.append(vocab[topi.item()])
            if topi.item() == 3:
                break
            decoder_input = topi
            decoder_init = decoder_hidden


        print("finish inferencing, using {} seconds".format(time.time()-time_start))
        print("".join(word_list))
        # print(topi)
        # print(decoder_hidden)
        print('\n')
    print("Done testing")


if __name__ == "__main__":
    
    main()


