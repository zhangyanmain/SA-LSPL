import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class MultiHeadSelfAtt(nn.Module):
    def __init__(self, n_heads, d_model):
        super(MultiHeadSelfAtt, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, val):
        bt_size = x.size(0)
        query = self.query(x)
        key = self.key(x)
        value = self.value(val)
        query = query.view(bt_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(bt_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(bt_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        att_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(att_weights, value)
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(bt_size, -1, self.d_model)
        output = self.fc(attended_values)
        output = output + val
        return output


class Inter_SelfAtt(nn.Module):
    def __init__(self, input_dim):
        super(Inter_SelfAtt, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        scores = torch.bmm(query, key.transpose(1, 2))
        att_weights = torch.softmax(scores, dim=2)
        value = self.value(x)
        weighted_value = torch.bmm(att_weights, value)
        return weighted_value


class MultiHeadSelfAtt_inter(nn.Module):
    def __init__(self, n_heads, d_model):
        super(MultiHeadSelfAtt_inter, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        bt_size = x.size(0)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        query = query.view(bt_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(bt_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(bt_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        att_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(att_weights, value)
        concatenated_values = attended_values.transpose(1, 2).contiguous().view(bt_size, -1, self.d_model)
        residual = x + concatenated_values
        output = self.fc(residual)
        output = self.dropout(output)
        return output


class Attn_user(nn.Module):
    def __init__(self, method, hidden_size, loc_unin_emb):
        super(Attn_user, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.loc_unin_Emb = loc_unin_emb
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.w1_socail = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.w2_socail = nn.Parameter(torch.tensor(0.4), requires_grad=True)

    def forward(self, user_emb, id_emb, socail_uid_emb):
        seq_len = id_emb.size()[0]
        state_len = user_emb.size()[0]
        socail_uid_emb = socail_uid_emb.unsqueeze(0)
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = self.w1_socail * self.score(user_emb[i], id_emb[j]) + self.w2_socail * self.score(
                    socail_uid_emb[i], id_emb[j])
        return F.softmax(attn_energies, dim=-1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


class Attn_loc(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn_loc, self).__init__()
        self.method = method  # dot
        self.hidden_size = hidden_size  # 1
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, his, cur, poi_distance_mat):
        seq_len = len(his)
        state_len = len(cur)
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()  # [10, 11]
        for i in range(state_len):
            for j in range(seq_len):
                if poi_distance_mat[cur[i], his[j]] != 0:
                    attn_energies[i, j] = 1 / poi_distance_mat[cur[i], his[j]]
                else:
                    attn_energies[i, j] = 1e-6
        return F.softmax(attn_energies, dim=-1)  # [10, 11]


class Attn_time(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn_time, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, his, cur, time_sim_mat):
        seq_len = len(his)  # 11
        state_len = len(cur)  # 10
        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda()  # [10, 11]
        for i in range(state_len):
            for j in range(seq_len):
                attn_energies[i, j] = time_sim_mat[cur[i], his[j]]
        return F.softmax(attn_energies, dim=-1)  # [10, 11]


class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, cur_session_represent, embedded_tensor_time, embedded_tensor_distance):
        delta = torch.cat([embedded_tensor_time, embedded_tensor_distance], dim=-1)  # 10, 10,100
        delta = torch.sum(delta, dim=-1)
        delta = torch.squeeze(delta, dim=-1)
        delta = delta.unsqueeze(0).cuda()  # 1,10,10
        query = self.query(cur_session_represent.unsqueeze(0))
        key = self.key(cur_session_represent.unsqueeze(0)).transpose(-1, -2)
        attn = torch.add(torch.bmm(query, key), delta)
        attn = F.softmax(attn, dim=-1)  # 1,10,10
        value = self.value(cur_session_represent.unsqueeze(0))  # 1,10,500
        attn_out = torch.bmm(attn.float(), value.float())
        out = torch.squeeze(attn_out, 0)
        return out


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def traj_att(query, key, value, sim=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = torch.nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SALSPL(nn.Module):
    def __init__(self, parameters, data_neural, weight=None, weight_cid=None, weight_user=None):
        super(self.__class__, self).__init__()
        self.n_users = parameters.uid_size
        self.n_items = parameters.loc_size
        self.hidden_units = parameters.hidden_size
        self.emb_size = parameters.hidden_size
        self.emb_size_his = 250
        self.item_emb = nn.Embedding.from_pretrained(weight, freeze=True)
        self.emb_tim = nn.Embedding(49, parameters.tim_emb_size, pding_idx=48)
        self.emb_user = nn.Embedding.from_pretrained(weight_user, freeze=True)
        self.emb_weeks = nn.Embedding(9, 10, pding_idx=8)
        self.emb_cat = nn.Embedding.from_pretrained(weight_cid, freeze=True, pding_idx=0)
        self.emb_unin_loc = nn.Embedding(2, 40)
        self.emb_unin_loc.weight.requires_grad = False
        self.loc_unin_emb = self.emb_unin_loc(torch.tensor([1]))
        self.his_input_size = self.emb_size + 10 + 50 + 10
        self.lstmcell = nn.LSTM(input_size=self.his_input_size,
                                hidden_size=parameters.hidden_size)  # LSTM(500, 500)
        self.lstmcell_his = nn.LSTM(self.his_input_size, self.emb_size_his, 1, bidirectional=True)
        self.linear = nn.Linear(parameters.hidden_size * 2,
                                parameters.loc_size)
        self.dlt_ = nn.LSTMCell(input_size=parameters.hidden_size,
                                       hidden_size=parameters.hidden_size)
        self.linear1 = nn.Linear(parameters.hidden_size,
                                 parameters.hidden_size)
        self.linear_2 = nn.Linear(parameters.hidden_size * 2,
                                  parameters.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.data_neural = data_neural
        self.attn_user = Attn_user('dot', self.emb_size, self.loc_unin_emb).cuda()
        self.attn_loc = Attn_loc('dot', 1).cuda()  # (dot,1)
        self.attn_time = Attn_time('dot', 1).cuda()  # (dot,1)
        self.SelfAttn = SelfAttn(parameters.hidden_size, parameters.hidden_size).cuda()
        self.SelfAtt_inter = Inter_SelfAtt(parameters.hidden_size).cuda()
        self.fc_user = nn.Linear(500, self.emb_size)  # (40,500)
        self.fc_weeks = nn.Linear(10, self.emb_size)  # (10,500)
        self.w1_out = nn.Parameter(torch.tensor(parameters.w1_out), requires_grad=True)
        self.w2_out = nn.Parameter(torch.tensor(parameters.w2_out), requires_grad=True)
        self.w1_dlt = nn.Parameter(torch.tensor(parameters.w1_dlt), requires_grad=True)
        self.w2_dlt = nn.Parameter(torch.tensor(parameters.w2_dlt), requires_grad=True)
        self.w3_dlt = nn.Parameter(torch.tensor(parameters.w3_dlt), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def create_dlt_input(self, session_seq_cur_bt, session_seq_cate_cur_bt, poi_dis_sig, tim_mat,
                             cat_trans):
        seqs_dlt_input_bt = []
        for i in range(len(session_seq_cur_bt)):
            seq_l = len(session_seq_cur_bt[i])
            session_seq_cur_bt[i].reverse()
            session_seq_cate_cur_bt[i].reverse()
            session_dlt__input_ind = [0] * seq_l
            for k in range(seq_l):
                cur_poi = [session_seq_cur_bt[i][k]]
                cur_poi_cat = [session_seq_cate_cur_bt[i][k]]
                poi_before = session_seq_cur_bt[i][k + 1:]
                poi_before_cat = session_seq_cate_cur_bt[i][k + 1:]
                distance_row = poi_dis_sig[cur_poi]
                distance_row_explicit = distance_row[:, poi_before][0]
                time_row = tim_mat[cur_poi]
                time_row_explicit = time_row[:, poi_before][0]
                cat_row = cat_trans[cur_poi_cat]
                cat_row_explicit = cat_row[:, poi_before_cat][0]
                w1 = self.w1_dlt
                w2 = self.w2_dlt
                w3 = self.w3_dlt
                min_score = np.inf
                min_ind = -1
                if np.all(np.isinf(time_row_explicit)) or np.all(cat_row_explicit == 0):
                    for ind in range(len(distance_row_explicit)):
                        score = distance_row_explicit[ind]
                        if score < min_score:
                            min_score = score
                            min_ind = ind
                else:
                    for ind in range(len(distance_row_explicit)):
                        if np.isinf(time_row_explicit[ind]):
                            score = w1 / (w1 + w3) * distance_row_explicit[ind] + w3 / (w1 + w3) * \
                                    cat_row_explicit[ind]
                        else:
                            score = w1 / (w1 + w2 + w3) * distance_row_explicit[ind] + w2 / (w1 + w2 + w3) * \
                                    time_row_explicit[ind] + w3 / (w1 + w2 + w3) * \
                                    cat_row_explicit[ind]
                        if score < min_score:
                            min_score = score
                            min_ind = ind
                ind_closet = min_ind
                session_dlt__input_ind[seq_l - k - 1] = seq_l - 2 - ind_closet - k
            session_seq_cur_bt[i].reverse()
            session_seq_cate_cur_bt[i].reverse()
            seqs_dlt_input_bt.append(session_dlt__input_ind)
        return seqs_dlt_input_bt

    def forward(self, user_vectors, item_vectors, seq_bt_tgt, mask_bt_ix_non_local, session_id_bt,
                seq_tim_bt, pd_seq_bt_tim, pd_seq_bt_week, pd_seq_bt_cate, is_train,
                poi_distance_mat, time_sim_mat, seq_bt, seq_cate_bt, poi_dis_sig,
                cat_trans_sig, user_sim):
        bt_size = item_vectors.size()[0]
        seq_size = item_vectors.size()[1]
        items = self.item_emb(item_vectors)
        tgt_embed = self.item_emb(seq_bt_tgt)
        items_time = self.emb_tim(pd_seq_bt_tim)
        items_week = self.emb_weeks(pd_seq_bt_week)
        items_cat = self.emb_cat(pd_seq_bt_cate)
        items_new = torch.cat((items, items_time, items_week, items_cat), dim=2)
        items_new = self.dropout(items_new)
        x1 = items
        item_vectors = item_vectors.cpu()
        x = items_new
        x = x.transpose(0, 1)
        h1 = Variable(torch.zeros(1, bt_size, self.hidden_units)).cuda()
        c1 = Variable(torch.zeros(1, bt_size, self.hidden_units)).cuda()
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        out = out.transpose(0, 1)  # 11 500
        user_bt = np.array(user_vectors.cpu())
        y_list = []
        out_hie = []
        out_new = []
        seq_dlt_bt = self.create_dlt_input(seq_bt, seq_cate_bt, poi_dis_sig, cat_trans_sig)
        for ii in range(bt_size):
            cur_input_dlt_ind = seq_dlt_bt[ii]
            hiddens_cur = x1[ii]
            dlt_lstm_outs_h = []
            dlt_lstm_outs_c = []
            for ind_dlt in range(len(cur_input_dlt_ind)):
                ind_dlt_explicit = cur_input_dlt_ind[ind_dlt]
                hidden_cur = hiddens_cur[ind_dlt].unsqueeze(0)
                if ind_dlt == 0:
                    h = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    c = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    (h, c) = self.dlt_(hidden_cur, (h, c))
                    dlt_lstm_outs_h.append(h)
                    dlt_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dlt_(hidden_cur, (
                        dlt_lstm_outs_h[ind_dlt_explicit], dlt_lstm_outs_c[ind_dlt_explicit]))
                    dlt_lstm_outs_h.append(h)
                    dlt_lstm_outs_c.append(c)
            dlt_lstm_outs_h.append(hiddens_cur[len(cur_input_dlt_ind):])
            dlt_out = torch.cat(dlt_lstm_outs_h, dim=0).unsqueeze(0)
            out_hie.append(dlt_out)
            cur_session_timid = seq_tim_bt[ii]
            cur_session_poiid = item_vectors[ii][:len(cur_session_timid)]
            session_id_cur = session_id_bt[ii]
            cur_session_embed = out[ii]
            cur_session_mask = mask_bt_ix_non_local[ii].unsqueeze(1)
            seq_l = int(sum(np.array(cur_session_mask.cpu()))[0])
            cur_session_represent_list = []
            if is_train:
                for iii in range(seq_l):
                    cur_session_represent = torch.sum(cur_session_embed * cur_session_mask,
                                                          dim=0).unsqueeze(0) / sum(cur_session_mask)
                    cur_session_represent_list.append(cur_session_represent)
            else:
                for iii in range(seq_l):
                    cur_session_represent_rep_item = cur_session_embed[0:iii + 1]
                    cur_session_represent_rep_item = torch.sum(cur_session_represent_rep_item, dim=0).unsqueeze(
                        0) / (iii + 1)
                    cur_session_represent_list.append(cur_session_represent_rep_item)
            cur_session_represent = torch.cat(cur_session_represent_list, dim=0)
            cur_session_represent_ht_1 = cur_session_represent
            list_for_sessions = []
            h2 = Variable(torch.zeros(2, 1, 250)).cuda()
            c2 = Variable(torch.zeros(2, 1, 250)).cuda()
            user_id_cur = user_bt[ii]
            similarity_vector_a = user_sim[user_id_cur]
            max_similarity = -1
            social_sim_uid = -1
            for user_ind, similarity_value in enumerate(similarity_vector_a):
                if user_ind != user_id_cur and similarity_value > max_similarity:
                    max_similarity = similarity_value
                    social_sim_uid = user_ind
            for jj in range(session_id_cur):
                seq = [s[0] for s in self.data_neural[user_id_cur]['sessions'][jj]]
                se_bac = seq
                seq = Variable(torch.LongTensor(np.array(seq))).cuda()
                se_emb = self.item_emb(seq).unsqueeze(1)  # 11,1,500
                self.item_emb(seq).unsqueeze(1)
                seq_week = [s[4] for s in self.data_neural[user_id_cur]['sessions'][jj]]
                seq_week = Variable(torch.LongTensor(np.array(seq_week))).cuda()
                seq_week_emb = self.emb_weeks(seq_week).unsqueeze(1)  # 11,1,500
                seq_tim_id = [s[1] for s in self.data_neural[user_id_cur]['sessions'][jj]]
                seq_tim_id_bac = seq_tim_id
                seq_tim_id = Variable(torch.LongTensor(np.array(seq_tim_id))).cuda()
                seq_tim_id_emb = self.emb_tim(seq_tim_id).unsqueeze(1)  # 11,1,500
                seq_cat = [s[2] for s in self.data_neural[user_id_cur]['sessions'][jj]]
                seq_cat = Variable(torch.LongTensor(np.array(seq_cat))).cuda()
                seq_cat_emb = self.emb_cat(seq_cat).unsqueeze(1)  # 11,1,500
                se_emb = torch.cat(
                    (se_emb, seq_tim_id_emb, seq_week_emb, seq_cat_emb),
                    dim=2)  # 11,1,1000
                se_emb = self.dropout(se_emb)
                se_emb, (h2, c2) = self.lstmcell_his(se_emb, (h2, c2))
                attn_weights_loc = self.attn_loc(se_bac, cur_session_poiid, poi_distance_mat)
                attn_weights_time = self.attn_time(seq_tim_id_bac, cur_session_timid, time_sim_mat)
                attn_weights_loc = attn_weights_loc.unsqueeze(0)
                attn_weights_time = attn_weights_time.unsqueeze(0)
                se_emb = se_emb.transpose(0, 1)
                context_loc = attn_weights_loc.bmm(se_emb).squeeze(0)
                context_time = attn_weights_time.bmm(se_emb).squeeze(0)
                his_c_loc_tim = context_loc + context_time
                his_c_loc_tim = his_c_loc_tim.squeeze(0)
                seq_uid = torch.tensor([user_bt[ii]]).cuda()
                uid_emb = self.emb_user(seq_uid)
                uid_emb = self.fc_user(uid_emb)
                socail_uid_tensor = torch.tensor(social_sim_uid).cuda()
                socail_uid_emb = self.emb_user(socail_uid_tensor)
                socail_uid_emb = self.fc_user(socail_uid_emb)
                user_attn_weights = self.attn_user(uid_emb, his_c_loc_tim, socail_uid_emb).unsqueeze(
                    0)
                his_c_loc_tim = user_attn_weights.bmm(his_c_loc_tim.unsqueeze(0)).squeeze(
                    0)
                context_user = his_c_loc_tim.repeat(seq_l, 1).unsqueeze(
                    0)
                list_for_sessions.append(context_user)
            his_intra_trajectory = torch.cat(list_for_sessions, dim=0)
            his_intra_trajectory = self.SelfAtt_inter(his_intra_trajectory)
            cur_session_embed_new_all = cur_session_embed.unsqueeze(0) + uid_emb.repeat(
                cur_session_embed.unsqueeze(0).shape[1], 1).unsqueeze(0)
            out_new.append(cur_session_embed_new_all)
            sessions_represent = his_intra_trajectory.transpose(0, 1).cuda()
            cur_session_represent_ht_1 = cur_session_represent_ht_1.unsqueeze(
                2).cuda()
            sims = F.softmax(sessions_represent.bmm(cur_session_represent_ht_1).squeeze(2), dim=1).unsqueeze(
                1)
            out_y_cur = torch.selu(self.linear1(sims.bmm(sessions_represent).squeeze(1)))
            out_y_cur_pd = torch.zeros(seq_size - seq_l, self.emb_size).cuda()
            out_layer_2_list = [out_y_cur, out_y_cur_pd]
            out_layer_2 = torch.cat(out_layer_2_list, dim=0).unsqueeze(0)
            y_list.append(out_layer_2)
        y = torch.selu(torch.cat(y_list,
                                 dim=0))
        out_hie = F.selu(torch.cat(out_hie, dim=0))
        out = F.selu(torch.cat(out_new, dim=0))
        out = self.w1_out / (self.w1_out + self.w2_out) * out + self.w2_out / (
                self.w1_out + self.w2_out) * out_hie
        out_put_emb_v1 = torch.cat([y, out], dim=2)
        out_put_emb_v1 = self.dropout(out_put_emb_v1)
        hiddenstate = self.linear_2(out_put_emb_v1)
        output_ln = self.linear(out_put_emb_v1)
        output = F.log_softmax(output_ln, dim=-1)
        return output, hiddenstate, tgt_embed
