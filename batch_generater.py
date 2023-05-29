import torch
from utils import fix_length
from einops import rearrange


def cf_train_quadkey(batch, data_source, max_len, sampler, quadkey_processor, TIME_processor, loc2quadkey, num_neg):
    # src_seq, trg_seq, t_mat, g_mat = zip(*batch)
    src_seq, trg_seq = zip(*batch)
    # t_mat_ = torch.stack(t_mat)
    # g_mat_ = torch.stack(g_mat)
    src_user_, src_locs_, src_quadkeys_, src_timecode_, src_lat_, src_lng_, src_times_ = [
    ], [], [], [], [], [], []
    data_size = []
    for e in src_seq:
        u_, l_, q_, t_, lat_, lng_, tg_, _ = zip(*e)
        src_user_.append(torch.tensor(u_))
        src_lat_.append(torch.tensor(lat_))
        src_lng_.append(torch.tensor(lng_))
        data_size.append(len(u_))
        src_locs_.append(torch.tensor(l_))
        q_ = quadkey_processor.numericalize(list(q_))
        # tg_ = TIME_processor.numericalize(list(tg_))
        src_timecode_.append(torch.tensor(tg_))
        src_quadkeys_.append(q_)
        src_times_.append(torch.tensor(t_))
    src_user_ = fix_length(src_user_, 1, max_len, 'train src seq')

    src_lat_ = fix_length(src_lat_, 1, max_len, 'train src seq')
    src_lng_ = fix_length(src_lng_, 1, max_len, 'train src seq')
    src_locs_ = fix_length(src_locs_, 1, max_len, 'train src seq')
    src_quadkeys_ = fix_length(src_quadkeys_, 2, max_len, 'train src seq')
    src_timecode_ = fix_length(src_timecode_, 2, max_len, 'train src seq')
    src_times_ = fix_length(src_times_, 1, max_len, 'train src seq')

    trg_locs_ = []
    trg_quadkeys_ = []
    trg_time_grams_ = []
    trg_times_ = []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        pos_time_grams = torch.tensor([[e[6]] for e in seq])
        trg_times =  torch.tensor([[e[3]] for e in seq])
        neg = sampler(seq, num_neg, user=seq[0][0])
        pos_neg_locs = torch.cat([pos, neg], dim=-1)
        # pos_time_grams = pos_time_grams.repeat(1, 1+num_neg, 1)
        
        trg_times_.append(trg_times)
        trg_locs_.append(pos_neg_locs)
        trg_time_grams_.append(pos_time_grams)
        pos_neg_quadkey = []
        for l in range(pos_neg_locs.size(0)):
            q_key = []
            for loc_idx in pos_neg_locs[l]:
                q_key.append(loc2quadkey[loc_idx])
            pos_neg_quadkey.append(quadkey_processor.numericalize(q_key))
        # print(pos_neg_quadkey[0].size())
        # exit(0)
        trg_quadkeys_.append(torch.stack(pos_neg_quadkey))
    trg_locs_ = fix_length(trg_locs_, n_axies=2,
                           max_len=max_len, dtype='train trg seq')
    # print(trg_quadkeys_[0].size())
    # exit(0)
    trg_times_ = fix_length(trg_times_, n_axies=2,
                           max_len=max_len, dtype='train trg seq')
    trg_times_ = rearrange(
        rearrange(trg_times_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')

    trg_time_grams_ = fix_length(
        trg_time_grams_, n_axies=3, max_len=max_len, dtype='train trg seq')
    trg_time_grams_ = rearrange(rearrange(
        trg_time_grams_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    trg_locs_ = rearrange(
        rearrange(trg_locs_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    
    trg_quadkeys_ = fix_length(
        trg_quadkeys_, n_axies=3, max_len=max_len, dtype='train trg seq')
    trg_quadkeys_ = rearrange(rearrange(
        trg_quadkeys_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    # print(trg_time_grams_.size())
    # exit(0)
    # print(src_locs_.size(),src_quadkeys_.size(),trg_locs_.size(),trg_quadkeys_.size())

    # return src_locs_, src_quadkeys_, src_times_, t_mat_, g_mat_, trg_locs_, trg_quadkeys_, data_size
    return src_user_, src_locs_, src_quadkeys_, src_times_, src_timecode_, src_lat_, src_lng_, trg_locs_, trg_quadkeys_,trg_times_, trg_time_grams_, data_size


def cf_eval_quadkey(batch, data_source, max_len, sampler, quadkey_processor, timestamp_processor, loc2quadkey, num_neg):
    # src_seq, trg_seq, t_mat, g_mat = zip(*batch)
    # t_mat_ = torch.stack(t_mat)
    # g_mat_ = torch.stack(g_mat)
    src_seq, trg_seq = zip(*batch)
    src_user_, src_locs_, src_quadkeys_, src_timecode_, src_lat_, src_lng_, src_times_ = [
    ], [], [], [], [], [], []
    # src_locs_, src_quadkeys_, src_times_ = [], [], []
    data_size = []
    for e in src_seq:
        u_, l_, q_, t_, lat_, lng_, tg_, _ = zip(*e)
        src_user_.append(torch.tensor(u_))
        data_size.append(len(u_))
        src_locs_.append(torch.tensor(l_))
        src_lat_.append(torch.tensor(lat_))
        src_lng_.append(torch.tensor(lng_))
        q_ = quadkey_processor.numericalize(list(q_))
        # tg_ = timestamp_processor.numericalize(list(tg_))
        src_quadkeys_.append(q_)
        src_timecode_.append(torch.tensor(tg_))
        src_times_.append(torch.tensor(t_))
    src_user_ = fix_length(src_user_, 1, max_len, 'eval src seq')
    src_lat_ = fix_length(src_lat_, 1, max_len, 'eval src seq')
    src_lng_ = fix_length(src_lng_, 1, max_len, 'eval src seq')
    src_locs_ = fix_length(src_locs_, 1, max_len, 'eval src seq')
    src_quadkeys_ = fix_length(src_quadkeys_, 2, max_len, 'eval src seq')
    src_timecode_ = fix_length(src_timecode_, 2, max_len, 'eval src seq')
    src_times_ = fix_length(src_times_, 1, max_len, 'eval src seq')

    trg_locs_ = []
    trg_quadkeys_ = []
    trg_time_grams_ = []
    trg_times_ = []
    for i, seq in enumerate(trg_seq):
        pos = torch.tensor([[e[1]] for e in seq])
        pos_time_grams = torch.tensor([[e[6]] for e in seq])
        trg_times =  torch.tensor([[e[3]] for e in seq])
        neg_sample_from = [src_seq[i][-1]]
        # print(pos,neg_sample_from)
        # exit(0)
        neg = sampler(neg_sample_from, num_neg, user=neg_sample_from[0][0])
        # pos_time_grams = pos_time_grams.repeat(1, 1+num_neg, 1)
        # neg = sampler(seq, num_neg, user=seq[0][0])
        pos_neg_locs = torch.cat([pos, neg], dim=-1)

        trg_times_.append(trg_times)
        trg_locs_.append(pos_neg_locs)
        trg_time_grams_.append(pos_time_grams)
        pos_neg_quadkey = []
        for l in range(pos_neg_locs.size(0)):
            q_key = []
            for loc_idx in pos_neg_locs[l]:
                q_key.append(loc2quadkey[loc_idx])
            pos_neg_quadkey.append(quadkey_processor.numericalize(q_key))
        trg_quadkeys_.append(torch.stack(pos_neg_quadkey))

    trg_locs_ = fix_length(trg_locs_, n_axies=2,
                           max_len=max_len, dtype='eval trg loc')

    trg_times_ = fix_length(trg_times_, n_axies=2,
                           max_len=max_len, dtype='eval trg loc')
    trg_times_ = rearrange(
        rearrange(trg_times_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    
    trg_time_grams_ = fix_length(
        trg_time_grams_, n_axies=3, max_len=max_len, dtype='eval trg loc')
    trg_time_grams_ = rearrange(rearrange(
        trg_time_grams_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    trg_locs_ = rearrange(
        rearrange(trg_locs_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    trg_quadkeys_ = fix_length(
        trg_quadkeys_, n_axies=3, max_len=max_len, dtype='eval trg loc')
    trg_quadkeys_ = rearrange(rearrange(
        trg_quadkeys_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    # print(trg_time_grams_.size())
    # print(trg_quadkeys_.size())
    # exit(0)

    return src_user_, src_locs_, src_quadkeys_, src_times_, src_timecode_, src_lat_, src_lng_, trg_locs_, trg_quadkeys_,trg_times_, trg_time_grams_, data_size
