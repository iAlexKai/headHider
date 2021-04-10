import argparse
import time
from datetime import datetime
import numpy as np
import random
import json
import torch
import os, sys
import time
from torch2trt_dynamic import torch2trt_dynamic

from configs import Config as Config
from data_apis.corpus import LoadPoem
from models.poemwae import PoemWAE

from helper import to_tensor, timeSince  # 将numpy转为tensor
#
#
# parentPath = os.path.abspath("..")
# sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
#
# parser = argparse.ArgumentParser(description='headHider Pytorch')
#
# # 大古诗数据集
# parser.add_argument('--train_data_dir', type=str, default='./data/train_data.txt',
#                     help='addr of data corpus for train and valid')
#
# parser.add_argument('--test_data_dir', type=str, default='./data/test_data.txt',
#                     help='addr of data for testing, i.e. test titles')
#
# parser.add_argument('--max_vocab_size', type=int, default=10000, help='The size of the vocab, Cannot be None')
# parser.add_argument('--expname', type=str, default='basic',
#                     help='experiment name, for disinguishing different parameter settings')
# parser.add_argument('--model', type=str, default='WAE', help='name of the model')
# parser.add_argument('--visual', action='store_true', default=False, help='visualize training status in tensorboard')
# parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
# parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
#
# # Evaluation Arguments
# parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
# parser.add_argument('--log_every', type=int, default=50, help='interval to log training results')
# parser.add_argument('--valid_every', type=int, default=50, help='interval to validation')
# parser.add_argument('--eval_every', type=int, default=1, help='interval to evaluate on the validation set')
# parser.add_argument('--test_every', type=int, default=1, help='interval to test on the titles')
# parser.add_argument('--seed', type=int, default=1111, help='random seed')
# parser.add_argument('--forward_only', default=True, action='store_true', help='only test, no training')
#
# args = parser.parse_args()

# pretrain is True and test_align is False: 用五个小数据集训练从混合高斯分离出来的五个高斯分布，包含test
# pretrain is False and test_align is True: 测试大数据集训练出来的混合高斯中情感的align情况
# pretrain is False and test_align is False: 用大数据集训练5个高斯混合出来的模型

# if args.merge:
#     assert args.pretrain is False and args.test_align is False
# if args.divide:
#     assert args.pretrain is True and args.test_align is False

# if args.pretrain:
#     assert args.sentiment_path == '../final_data/poem_with_sentiment.txt'
#     assert args.test_align is False
#     assert args.dataset == 'SentimentPoems'

# if args.test_align:
#     assert args.dataset == 'TSPoems'
#     assert args.sentiment_path == '../final_data/poem_with_sentiment.txt'




# make output directory if it doesn't already exist

# Set the random seed manually for reproducibility.
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

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

    api = LoadPoem('./data/train_data.txt', './data/test_data.txt', 10000)

    model = PoemWAE(config=config, api=api)

    import time
    time_start = time.time()
    print("\nbefore loading model\n")

    model.load_state_dict(torch.load(f='./output/basic/model_state_dict.pckl'))
    model = model.cuda()
    print("finish loading model, using {:d} seconds".format(int(time.time()-time_start)))

    model.vocab = api.vocab
    model.rev_vocab = api.rev_vocab

    # 从test_loader里面拿到一个输入，使用tensorrt对模型进行压缩
    title_tensor = get_one_input_for_tensorrt(api.rev_vocab, config.title_size)

    # model = torch2trt_dynamic(model, [title_tensor], max_workspace_size=1 << 28)

    decoder_input = to_tensor(torch.IntTensor([[api.rev_vocab['<s>']]]).view(1, 1))  # (batch, 1)
    decoder_input = title_tensor
    model = torch2trt_dynamic(model, [title_tensor, decoder_input], max_workspace_size=1 << 28)

    last_title = None
    last_file_length = 0
    while True:
        model.eval()  # eval()主要影响BatchNorm, dropout等操作
        print("Waiting for vocal input")
        batch, last_file_length = get_user_input(api.rev_vocab, config.title_size, last_file_length)

            #batch = test_loader.next_batch_test()  # test data使用专门的batch
            #import pdb
            #pdb.set_trace()
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

        # epsilon = torch.randn([1, config.z_size]).cuda()
        # output_poem = model([title_tensor, epsilon])
        decoder_input = to_tensor(torch.IntTensor([[api.rev_vocab['<s>']]]).view(1, 1))  # (batch, 1)
        decoder_input = title_tensor
        output_poem = model(title_tensor, decoder_input)

        # output_poem = model([title_tensor])

        print("finish inferencing, using {} seconds".format(time.time()-time_start))

            #with open('./content_from_remote.txt', 'w') as file:
            #    file.write(output_poem)
        print(output_poem)
        print('\n')
    print("Done testing")


def train_process(model, train_loader, config, sentiment_data=False, mask_type=None):
    model.train()
    loss_records = []
    sentiment_mask = None
    if sentiment_data:

        batch = train_loader.next_sentiment_batch()
        finish_train = False
        if batch is None:  # end of epoch
            finish_train = True
            return model, finish_train, None
        title, context, target, target_lens, sentiment_mask = batch
        title, context, target, target_lens, sentiment_mask = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens), to_tensor(sentiment_mask)
    else:
        batch = train_loader.next_batch()
        finish_train = False
        if batch is None:  # end of epoch
            finish_train = True
            return model, finish_train, None
        title, context, target, target_lens = batch
        title, context, target, target_lens = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens)

    # import pdb
    # pdb.set_trace()
    loss_AE = model.train_AE(title, context, target, target_lens)  # 输入topic，last句，当前句，当前句长度
    loss_records.extend(loss_AE)

    loss_G = model.train_G(title, context, target, target_lens, sentiment_mask=sentiment_mask, mask_type=mask_type)
    loss_records.extend(loss_G)

    # 训练 Discriminator
    for i in range(config.n_iters_d):  # train discriminator/critic
        loss_D = model.train_D(title, context, target, target_lens)
        if i == 0:
            loss_records.extend(loss_D)
        if i == config.n_iters_d - 1:
            break
        batch = train_loader.sample_one_batch(sentiment=sentiment_data)
        if batch is None:  # end of epoch
            break
        title, context, target, target_lens = batch
        title, context, target, target_lens = \
            to_tensor(title), to_tensor(context), to_tensor(target), to_tensor(target_lens)

    return model, finish_train, loss_records



if __name__ == "__main__":
    
    main()


