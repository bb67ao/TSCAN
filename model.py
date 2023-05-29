import torch

from time_aware_pe import TAPE
from attn_modules import *


class STiSAN(nn.Module):
    def __init__(self, n_user, n_loc, n_quadkey, n_timestamp, features, exp_factor, k_t, k_d, depth, src_len, dropout, device):
        super(STiSAN, self).__init__()
        self.src_len = src_len
        self.device = device
        self.emb_loc = Embedding(n_loc, features, True, True)
        self.emb_quadkey = Embedding(n_quadkey, features, True, True)

        self.geo_encoder_layer = GeoEncoderLayer(features, exp_factor, dropout)
        self.geo_encoder = GeoEncoder(
            features, self.geo_encoder_layer, depth=2)

        self.timefeature = TimeFeatureEmbedding(5, features)

        self.k_t = torch.tensor(k_t)

        self.tscab = TSCAB(features, dropout)

        self.tiab = TIAB(features*2, exp_factor, dropout)

    def get_tmat_train(self, src_time, trg_time,k_t):
        max_len = self.src_len
        time_mat_i = trg_time.unsqueeze(-1).expand(
            [-1, max_len, max_len]).to(self.device)
        time_mat_j = src_time.unsqueeze(1).expand(
            [-1, max_len, max_len]).to(self.device)
        # day
        time_mat = torch.abs(time_mat_i - time_mat_j) / (3600. * 24)
        time_mat_max = (torch.ones_like(time_mat)*k_t)
        time_mat_ = torch.where(time_mat > time_mat_max,
                                time_mat_max, time_mat) - time_mat
        return time_mat_
    
    def get_tmat_eval(self, src_time,trg_time, k_t):
        max_len = self.src_len
        time_mat_i = src_time
        time_mat_j = trg_time.expand(
            [-1, max_len]).to(self.device)
        # day
        time_mat = torch.abs(time_mat_i - time_mat_j) / (3600. * 24)
        time_mat_max = (torch.ones_like(time_mat)*k_t)
        time_mat_ = torch.where(time_mat > time_mat_max,
                                time_mat_max, time_mat) - time_mat
        return time_mat_.unsqueeze(1)   


    def forward(self, src_user, src_loc, src_quadkey, src_time, src_timecodes, src_lat, src_lon, pad_mask, attn_mask,
                trg_loc, trg_quadkey, trg_time,trg_time_grams, key_pad_mask, mem_mask, ds,training):

        src_timecodes = src_timecodes.float()
        trg_time_grams = trg_time_grams.float()
        # (b, n, d)

        src_loc_emb = self.emb_loc(src_loc)

        src_quadkey_emb = self.emb_quadkey(src_quadkey)

        src_timefeature = self.timefeature(src_timecodes)

        # (b, n, d)
        src_quadkey_emb = self.geo_encoder(src_quadkey_emb)

        trg_quadkey_emb = self.emb_quadkey(trg_quadkey)

        trg_timefeature = self.timefeature(trg_time_grams)

        # (b, n, 2 * d)
        src = torch.cat([src_loc_emb, src_quadkey_emb], dim=-1)

        if training:trg_tmat = self.get_tmat_train(src_time,trg_time,self.k_t)
        else:trg_tmat = self.get_tmat_eval(src_time,trg_time,self.k_t)

        src_person = self.tscab(
            trg_timefeature, src_timefeature, src, attn_mask, pad_mask)

        if training:src = self.tiab(src_person, src, src, trg_tmat,attn_mask, pad_mask)
        else:src = self.tiab(src_person, src, src, trg_tmat,None,None)

        trg_loc_emb = self.emb_loc(trg_loc)
        trg_quadkey_emb = self.geo_encoder(trg_quadkey_emb)
        trg = torch.cat([trg_loc_emb, trg_quadkey_emb], dim=-1)

        # # (b, 1 + k)
        src = src.repeat(1, trg.size(1)//src.size(1), 1)
        output = torch.sum(src * trg, dim=-1)
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
