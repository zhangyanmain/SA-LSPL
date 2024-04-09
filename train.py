# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
from collections import deque, Counter

NYCdata = pickle.load(open('../data/NYC/FourSquareNYC.pk', 'rb'), encoding='iso-8859-1')
poi_dis_mat_sig = pickle.load(open('../data/NYC/dis_nyc_sig.pkl', 'rb'), encoding='iso-8859-1')
avg_time_mat_sig = pickle.load(open('../data/NYC/avg_time_mat_sig.pk', 'rb'), encoding='iso-8859-1')
cat_tran_sig = pickle.load(open('../data/NYC/cat_trans.pkl', 'rb'),
                           encoding='iso-8859-1')
user_sim_mat = pickle.load(open('../data/NYC/user_sim.pk', 'rb'), encoding='iso-8859-1')
data_neural = NYCdata['data_neural']
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, cid_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, w1_out=0.5, w2_out=0.5,
                 w1_dilated=0.6, w2_dilated=0.2, w3_dilated=0.2, w1_dilated_ST=0.6, w2_dilated_ST=0.4, w_s_acc=0.5,
                 optim='Adam', his_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 data_path='../data/NYC', save_path='../results/NYC', data_name='FourSquareNYC'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(open(self.data_path + self.data_name + '.pk', 'rb'), encoding='iso-8859-1')
        self.vid_list = data['vid_list']
        self.cid_list = data['cid_list']
        self.uid_list = data['uid_list']
        self.data_neural = data['data_neural']
        self.tim_size = 24
        self.loc_size = len(self.vid_list)
        self.cid_size = len(self.cid_list)
        self.uid_size = len(self.uid_list)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.cid_emb_size = cid_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size
        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip
        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.his_mode = his_mode
        self.model_mode = model_mode
        self.w1_out = w1_out
        self.w2_out = w2_out
        self.w1_dilated = w1_dilated
        self.w2_dilated = w2_dilated
        self.w3_dilated = w3_dilated
        self.w_s_acc = w_s_acc


def generate_input_his(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        train_id = data_neural[u][mode]
        data_train[u] = {}
        train_idx[u] = train_id
    return data_train, train_idx


def generate_input_long_his(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        train_id = data_neural[u][mode]
        data_train[u] = {}
        train_idx[u] = train_id
    return data_train, train_idx


def auxiliary(hidden, tgt_emb):
    seq_len = hidden.size()[0]
    loss_list = []
    for i in range(seq_len):
        res = hidden[i].dot(tgt_emb[i])
        res = F.sigmoid(res)
        loss_list.append(torch.log(1 - res + 1e-5) + torch.log(res + 1e-5))
    auxiliary_loss = - sum(loss_list) / seq_len
    return auxiliary_loss


def generate_detailed_bt_data(one_train_bt):
    session_id_bt = []
    user_id_bt = []
    seq_bt = []
    seqs_lens_bt = []
    seqs_tim_bt = []
    seqs_week_bt = []
    seq_categories_bt = []
    seq_tgt_bt = []
    for sample in one_train_bt:
        user_id_bt.append(sample[0])
        session_id_bt.append(sample[1])
        session_seq_cur = [s[0] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_seq_tim_cur = [s[1] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_seq_week_cur = [s[4] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_seq_cat_cur = [s[2] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        seq_bt.append(session_seq_cur[:-1])
        seqs_lens_bt.append(len(session_seq_cur[:-1]))
        seqs_tim_bt.append(session_seq_tim_cur[:-1])
        seqs_week_bt.append(session_seq_week_cur[:-1])
        seq_categories_bt.append(session_seq_cat_cur[:-1])
        seq_tgt_bt.append(session_seq_cur[1:])
    return user_id_bt, session_id_bt, seq_bt, seqs_lens_bt, seqs_tim_bt, seqs_week_bt, seq_categories_bt, seq_tgt_bt


def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())
    train_queue = list()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def get_acc(tgt, scores):
    tgt = tgt.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = tgt[i]
        if t != 0:
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return acc, ndcg


def get_hint(tgt, scores, users_visited):
    tgt = tgt.data.cpu().numpy()
    val, idxx = scores.data.topk(1, 1)
    predx = idxx.cpu().numpy()
    hint = np.zeros((3,))
    count = np.zeros((3,))
    count[0] = len(tgt)
    for i, p in enumerate(predx):
        t = tgt[i]
        if t == p[0] and t > 0:
            hint[0] += 1
        if t in users_visited:
            count[1] += 1
            if t == p[0] and t > 0:
                hint[1] += 1
        else:
            count[2] += 1
            if t == p[0] and t > 0:
                hint[2] += 1
    return hint, count


def auxiliary(hidden, tgt_emb):
    bt_size = hidden.size(0)
    seq_len = hidden.size(1)
    loss_list = []
    for i in range(bt_size):
        bt_loss = 0
        for j in range(seq_len):
            res = hidden[i, j].dot(tgt_emb[i, j])
            res = torch.sigmoid(res)
            bt_loss += torch.log(1 - res + 1e-5) + torch.log(res + 1e-5)
        loss_list.append(bt_loss / seq_len)
    auxiliary_loss = -sum(loss_list) / bt_size
    return auxiliary_loss


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        size = output1.shape[0]
        euclidean_dis = F.pairwise_dis(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_dis, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_dis, min=0.0), 2))
        return loss_contrastive / size


def pad_bt_of_lists_masks(bt_of_lists, pd_seq_bt_time, pd_seq_bt_week,
                          pd_seq_bt_cat, max_len):
    pd = [l + [0] * (max_len - len(l)) for l in bt_of_lists]
    pd_mask = [[1.0] * (len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in bt_of_lists]
    padde_mask_non_local = [[1.0] * (len(l)) + [0.0] * (max_len - len(l)) for l in bt_of_lists]
    pd_seq_bt_time = [l + [48] * (max_len - len(l)) for l in pd_seq_bt_time]
    pd_seq_bt_week = [l + [8] * (max_len - len(l)) for l in pd_seq_bt_week]
    pd_seq_bt_cat = [l + [373] * (max_len - len(l)) for l in pd_seq_bt_cat]
    return pd, pd_seq_bt_time, pd_seq_bt_week, pd_seq_bt_cat, pd_mask, padde_mask_non_local


def minibt(*tensors, **kwargs):
    bt_size = kwargs.get('bt_size', 128)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), bt_size):
            yield tensor[i:i + bt_size]
    else:
        for i in range(0, len(tensors[0]), bt_size):
            yield tuple(x[i:i + bt_size] for x in tensors)


def run_simple(data, run_idx, auxiliary_rate, mode, lr, clip, model,
               optimizer, criterion, mode2=None, time_sim_mat=None,
               poi_dis_mat=None):
    device = torch.device("cuda")
    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'random', 'test')
    total_loss = []
    queue_len = len(run_queue)
    print(queue_len)
    time_sim_mat = time_sim_mat
    poi_dis_mat = poi_dis_mat
    users_acc = {}
    bt_size = 32
    for one_train_bt in minibt(run_queue, bt_size=bt_size):
        torch.cuda.empty_cache()
        user_id_bt, session_id_bt, seq_bt, seqs_lens_bt, seq_tim_bt, seq_week_bt, \
            seq_categories_bt, seq_tgt_bt = generate_detailed_bt_data(one_train_bt)
        max_len = max(seqs_lens_bt)
        pd_seq_bt, pd_seq_bt_time, pd_seq_bt_week, pd_seq_bt_cat, \
            mask_bt_ix, mask_bt_ix_non_local = pad_bt_of_lists_masks \
            (seq_bt, seq_tim_bt, seq_week_bt, seq_categories_bt, max_len)
        seq_tgt_bt = [sublist + [0] * (max_len - len(sublist)) for sublist in seq_tgt_bt]
        seq_tgt_bt = Variable(torch.LongTensor(np.array(seq_tgt_bt))).to(device)
        pd_seq_bt = Variable(torch.LongTensor(np.array(pd_seq_bt))).to(device)
        pd_seq_bt_time = Variable(torch.LongTensor(np.array(pd_seq_bt_time))).to(device)
        pd_seq_bt_week = Variable(torch.LongTensor(np.array(pd_seq_bt_week))).to(
            device)
        pd_seq_bt_cat = Variable(torch.LongTensor(np.array(pd_seq_bt_cat))).to(
            device)
        mask_bt_ix = Variable(torch.FloatTensor(np.array(mask_bt_ix))).to(device)
        mask_bt_ix_non_local = Variable(torch.FloatTensor(np.array(mask_bt_ix_non_local))).to(device)
        user_id_bt = Variable(torch.LongTensor(np.array(user_id_bt))).to(device)
        logp_seq, hidden_state, tgt_embed = model(user_id_bt, pd_seq_bt, seq_tgt_bt,
                                                  mask_bt_ix_non_local, session_id_bt,seq_tim_bt,pd_seq_bt_time, pd_seq_bt_week,pd_seq_bt_cat, True, poi_dis_mat,
                                                  time_sim_mat,seq_bt, seq_categories_bt, poi_dis_mat_sig,avg_time_mat_sig,cat_tran_sig,user_sim_mat)
        predictions_logp = logp_seq[:, :] * mask_bt_ix[:, :, None]
        actual_next_tokens = seq_tgt_bt[:, :]  # 32 10
        logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])  # 32 10 1
        loss_poi = -logp_next.sum() / mask_bt_ix[:, :].sum()
        auxiliary_loss = auxiliary(hidden_state, tgt_embed)
        loss = loss_poi + auxiliary_rate * auxiliary_loss
        if mode == 'train':
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(p.grad.data, alpha=-lr)
            except:
                pass
            optimizer.step()
        elif mode == 'test':
            user_id_bt_test = user_id_bt
            for ii, u_cur in enumerate(user_id_bt_test):
                if u_cur not in users_acc:
                    users_acc[u_cur] = [0, 0, 0, 0, 0, 0, 0]
                acc, ndcg = get_acc(actual_next_tokens[ii], predictions_logp[ii])
                users_acc[u_cur][1] += acc[2][0]  # @1
                users_acc[u_cur][2] += acc[1][0]  # @5
                users_acc[u_cur][3] += acc[0][0]  # @10
                ###ndcg
                users_acc[u_cur][4] += ndcg[2][0]  # @1
                users_acc[u_cur][5] += ndcg[1][0]  # @5
                users_acc[u_cur][6] += ndcg[0][0]  # @10
                users_acc[u_cur][0] += (seqs_lens_bt[ii] - 1)
        total_loss.append(loss.data.cpu().numpy())
    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        tmp_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
            tmp_acc[2] = users_acc[u][3] + tmp_acc[2]

            tmp_acc[3] = users_acc[u][4] + tmp_acc[3]
            tmp_acc[4] = users_acc[u][5] + tmp_acc[4]
            tmp_acc[5] = users_acc[u][6] + tmp_acc[5]
            sum_test_samples = sum_test_samples + users_acc[u][0]
        avg_acc = (np.array(tmp_acc) / sum_test_samples).tolist()
        return avg_loss, avg_acc
