# coding: utf-8
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import time
from json import encoder

import gensim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
NYCdata = pickle.load(open('../data/NYC/FourSquareNYC.pk', 'rb'), encoding='iso-8859-1')
data_neural = NYCdata['data_neural']
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
wvmodel = gensim.models.KeyedVectors.load_word2vec_format("../data/NYC/loc_nyc.emb", binary=False,encoding='utf-8')
vocab_size = len(wvmodel.key_to_index) + 1
vector_size = wvmodel.vector_size
weight = torch.randn(vocab_size, vector_size)
word_to_idx = {}
idx_to_word = {}
idx = 0
for word, _ in wvmodel.key_to_index.items():
    word_to_idx[word] = idx
    idx_to_word[idx] = word
    idx += 1
for i in range(len(wvmodel.index_to_key)):
    try:
        index = word_to_idx[wvmodel.index_to_key[i]]
    except KeyError:
        continue
    vector = wvmodel.get_vector(wvmodel.index_to_key[i])
    weight[index, :] = torch.from_numpy(vector)
wvmodel = gensim.models.KeyedVectors.load_word2vec_format("../data/NYC/cat_nyc.emb", binary=False, encoding='utf-8')
vocab_size = len(wvmodel.key_to_index) + 1
vector_size = wvmodel.vector_size
weight_cid = torch.randn(vocab_size, vector_size)
word_to_idx = {}
idx_to_word = {}
idx = 1
for word, _ in wvmodel.key_to_index.items():
    word_to_idx[word] = idx
    idx_to_word[idx] = word
    idx += 1
word_to_idx['<unk>'] = 0
idx_to_word[0] = '<unk>'
for i in range(len(wvmodel.index_to_key)):
    try:
        index = word_to_idx[wvmodel.index_to_key[i]]
    except KeyError:
        continue
    vector = wvmodel.get_vector(wvmodel.index_to_key[i])
    weight_cid[index, :] = torch.from_numpy(vector)
from train import run_simple, RnnParameterData, generate_input_history, generate_input_long_history
from model import SALSPL


def run(args):
    parameters = RnnParameterData(loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  cid_emb_size=args.cid_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data_name=args.data_name, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, rnn_type=args.rnn_type,
                                  optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, w1_out=args.w1_out, w2_out=args.w2_out, w1_dilated=args.w1_dilated,
                                  w2_dilated=args.w2_dilated, w3_dilated=args.w3_dilated, w_s_acc=args.w_s_acc,
                                  epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, data_path=args.data_path, save_path=args.save_path)

    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'cid_emb_size': args.cid_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data_name': args.data_name, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip,
            'w1_out': args.w1_out, 'w2_out': args.w2_out,
            'w1_dilated': args.w1_dilated, 'w2_dilated': args.w2_dilated, 'w3_dilated': args.w3_dilated,
            'w_s_acc': args.w_s_acc,
            'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}

    print('*' * 15 + 'start training' + '*' * 15)
    print('users:{}'.format(parameters.uid_size))
    auxiliary_rate = 0.05
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = SALSPL(parameters=parameters, data_neural=data_neural, weight=weight, weight_cid=weight_cid).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-3)

    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': []}
    candidate = parameters.data_neural.keys()
    data_train, train_idx = generate_input_history(parameters.data_neural, 'train', candidate=candidate)
    data_test, test_idx = generate_input_long_history(parameters.data_neural, 'test', candidate=candidate)
    SAVE_PATH = args.save_path
    tmp_path = 'checkpoints/'
    if not SAVE_PATH + tmp_path:
        os.mkdir(SAVE_PATH + tmp_path)
    time_sim_matrix = pickle.load(open('../data/NYC/time_sim_NYC.pkl', 'rb'), encoding='iso-8859-1')
    poi_distance_matrix = pickle.load(open('../data/NYC/distance_nyc.pkl', 'rb'), encoding='iso-8859-1')
    torch.cuda.empty_cache()
    for epoch in range(parameters.epoch):
        st = time.time()
        print('epoch now： ', epoch)
        if args.pretrain == 0:
            model, avg_loss = run_simple(data_train, train_idx, auxiliary_rate, 'train', lr, parameters.clip, model,
                                         optimizer,
                                         criterion, parameters.model_mode, time_sim_matrix, poi_distance_matrix,
                                         )
            print('auxiliary_rate:{}'.format(auxiliary_rate))
            msg = 'auxiliary_rate:{}'.format(auxiliary_rate)
            with open(SAVE_PATH + "result.txt", "a") as file:
                file.write(msg + "\n")
            file.close()
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            msg = '==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr)
            with open(SAVE_PATH + "result.txt", "a") as file:
                file.write(msg + "\n")
            file.close()
            metrics['train_loss'].append(avg_loss)
        avg_loss, avg_acc = run_simple(data_test, test_idx, auxiliary_rate, 'test', lr, parameters.clip, model,
                                       optimizer, criterion, parameters.model_mode, time_sim_matrix,
                                       poi_distance_matrix)
        print(
            '==>Rec@1:{:.4f} Rec@5:{:.4f} Rec@10:{:.4f} NDCG@1:{:.4f} NDCG@5:{:.4f} NDCG@10:{:.4f} Loss:{:.4f}'.format(
                avg_acc[0], avg_acc[1], avg_acc[2], avg_acc[3], avg_acc[4], avg_acc[5], avg_loss))
        msg = '==>Rec@1:{:.4f} Rec@5:{:.4f} Rec@10:{:.4f} NDCG@1:{:.4f} NDCG@5:{:.4f} NDCG@10:{:.4f} Loss:{:.4f}'.format(
            avg_acc[0], avg_acc[1], avg_acc[2], avg_acc[3], avg_acc[4], avg_acc[5], avg_loss)
        with open(SAVE_PATH + "result.txt", "a") as file:
            file.write(msg + "\n")
        file.close()
        metrics['valid_loss'].append(avg_loss)
        metrics['accuracy'].append(avg_acc[0])
        save_name_tmp = 'ep_' + str(epoch) + '.m'
        torch.save(model.state_dict(), SAVE_PATH + save_name_tmp)
        scheduler.step(avg_acc[0])
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmax(metrics['accuracy'])
            load_name_tmp = 'ep_' + str(load_epoch) + '.m'
            model.load_state_dict(torch.load(SAVE_PATH + load_name_tmp))
            auxiliary_rate += 0.05
            print('load epoch={} model state'.format(load_epoch))
            msg = 'load epoch={} model state'.format(load_epoch)
            with open(SAVE_PATH + "result.txt", "a") as file:
                file.write(msg + "\n")
            file.close()
        print('single epoch time cost:{}'.format(time.time() - st))
        msg = 'single epoch time cost:{}'.format(time.time() - st)
        with open(SAVE_PATH + "result.txt", "a") as file:
            file.write(msg + "\n")
        file.close()
        if lr <= 0.9 * 1e-7:
            break
        if args.pretrain == 1:
            break
    mid = np.argmax(metrics['accuracy'])
    avg_acc = metrics['accuracy'][mid]
    load_name_tmp = 'ep_' + str(mid) + '.m'
    print("BEST_MODEL:", SAVE_PATH + load_name_tmp)
    return avg_acc


def load_pretrained_model(config):
    res = json.load(open("../pretrain/" + config.model_mode + "/res.txt"))
    args = Settings(config, res["args"])
    return args


class Settings(object):
    def __init__(self, config, res):
        self.data_path = config.data_path
        self.save_path = config.save_path
        self.data_name = res["data_name"]
        self.epoch_max = res["epoch_max"]
        self.learning_rate = res["learning_rate"]
        self.lr_step = res["lr_step"]
        self.lr_decay = res["lr_decay"]
        self.clip = res["clip"]
        self.dropout_p = res["dropout_p"]
        self.rnn_type = res["rnn_type"]
        self.attn_type = res["attn_type"]
        self.L2 = res["L2"]
        self.history_mode = res["history_mode"]
        self.model_mode = res["model_mode"]
        self.optim = res["optim"]
        self.hidden_size = res["hidden_size"]
        self.tim_emb_size = res["tim_emb_size"]
        self.loc_emb_size = res["loc_emb_size"]
        self.uid_emb_size = res["uid_emb_size"]
        self.voc_emb_size = res["cid_emb_size"]
        self.pretrain = 1


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--cid_emb_size', type=int, default=50, help="cid embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data_name', type=str, default='FourSquareNYC')
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)  # 5 * 1e-4
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-3, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--w1_out', type=float, default=0.5)
    parser.add_argument('--w2_out', type=float, default=0.5)
    parser.add_argument('--w1_dilated', type=float, default=0.6)
    parser.add_argument('--w2_dilated', type=float, default=0.2)
    parser.add_argument('--w3_dilated', type=float, default=0.2)
    parser.add_argument('--w_s_acc', type=float, default=0.5)
    parser.add_argument('--epoch_max', type=int, default=80)
    parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])
    parser.add_argument('--data_path', type=str, default='../data/NYC/')
    parser.add_argument('--save_path', type=str, default='../results/NYC')
    parser.add_argument('--model_mode', type=str, default='attn_avg_long_user',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    if args.pretrain == 1:
        args = load_pretrained_model(args)
    ours_acc = run(args)
