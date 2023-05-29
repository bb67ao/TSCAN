import time

import torch
from einops import rearrange
from tqdm import tqdm
from evaluator import evaluate
from batch_generater import cf_train_quadkey
from utils import *
from torch.utils.data import DataLoader


def train(model, max_len, train_data, train_sampler, train_bsz, train_num_neg, num_epoch, quadkey_processor, timestamp_processor,loc2quadkey,
          eval_data, eval_sampler, eval_bsz, eval_num_neg, optimizer, loss_fn, device, num_workers, log_path, result_path, model_path, parallel):
    for epoch_idx in range(num_epoch):
        start_time = time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_data,
                                 sampler=LadderSampler(train_data, train_bsz),
                                 num_workers=num_workers, batch_size=train_bsz,
                                 collate_fn=lambda e: cf_train_quadkey(
                                     e,
                                     train_data,
                                     max_len,
                                     train_sampler,
                                     quadkey_processor,
                                     timestamp_processor,
                                     loc2quadkey,
                                     train_num_neg))
        print("=====epoch {:>2d}=====".format(epoch_idx))
        batch_iterator = tqdm(enumerate(data_loader),
                              total=len(data_loader), leave=True,colour='blue')
        # if not any(enumerate(data_loader)):
        #     print("DataLoader is empty")
        model.train()
        for batch_idx, (src_user_,src_locs_, src_quadkeys_, src_times_,src_timecodes, lat_, lng_, trg_locs_, trg_quadkeys_, trg_times_,trg_time_grams_,data_size) in batch_iterator:
            src_loc = src_locs_.to(device)
            src_user = src_user_.to(device)
            src_quadkey = src_quadkeys_.to(device)
            src_timecodes = src_timecodes.to(device)
            src_time = src_times_.to(device)
            trg_time_grams = trg_time_grams_.to(device)
            # t_mat = t_mat_.to(device)
            # g_mat = g_mat_.to(device)
            lat = lat_.to(device)
            lng = lng_.to(device)
            trg_loc = trg_locs_.to(device)
            trg_times = trg_times_.to(device)
            trg_quadkey = trg_quadkeys_.to(device)
            pad_mask = get_pad_mask(data_size, max_len, device)
            attn_mask = get_attn_mask(max_len, device)
            mem_mask = get_mem_mask(max_len, train_num_neg, device)
            key_pad_mask = get_key_pad_mask(
                data_size, max_len, train_num_neg, device)
            optimizer.zero_grad()
            output = model(src_user,src_loc, src_quadkey, src_time, src_timecodes,lat, lng, pad_mask, attn_mask,
                           trg_loc, trg_quadkey,trg_times,trg_time_grams, key_pad_mask, mem_mask, data_size,True)
            output = rearrange(
                rearrange(output, 'b (k n) -> b k n', k=1 + train_num_neg), 'b k n -> b n k')
            pos_scores, neg_scores = output.split([1, train_num_neg], -1)
            loss = loss_fn(pos_scores, neg_scores) 
            keep = [torch.ones(e, dtype=torch.float32).to(device)
                    for e in data_size]
            keep = fix_length(keep, 1, max_len, dtype="exclude padding term")

            loss = torch.sum(loss * keep) / \
                torch.sum(torch.tensor(data_size).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss.item():.4f}")

        epoch_time = time.time() - start_time
        cur_avg_loss = running_loss / processed_batch
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))

        hr, ndcg = evaluate(model, max_len, eval_data, eval_sampler, eval_bsz,
                            eval_num_neg, quadkey_processor,timestamp_processor, loc2quadkey, device, num_workers)

        # save train log
        log_file = open(log_path, 'a+', encoding='utf8')
        print("epoch={:d}, loss={:.4f}".format(
            epoch_idx + 1, cur_avg_loss), file=log_file)
        print("Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(
            hr[4], ndcg[4], hr[9], ndcg[9]), file=log_file)
        log_file.close()

        # save model
        save_path = model_path + "H@5:{:.2f}N@5:{:.2f}H@10{:.2f}N@10:{:.2f}".format(
            hr[4], ndcg[4], hr[9], ndcg[9])
        save_path = save_path + '.pth'
        if parallel:
            torch.save(model.module.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

    print("training completed!")
    print("")
    print("=====evaluation under sampled metric (100 nearest un-visited locations)=====")
    hr, ndcg = evaluate(model, max_len, eval_data, eval_sampler, eval_bsz,
                        eval_num_neg, quadkey_processor,timestamp_processor, loc2quadkey, device, num_workers)
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(
        hr[4], ndcg[4], hr[9], ndcg[9]))
    # save result
    result_file = open(result_path, 'a+')
    print("Hit@5: {:.4f}, NDCG@5: {:.4f}, Hit@10: {:.4f}, NDCG@10: {:.4f} ".format(
        hr[4], ndcg[4], hr[9], ndcg[9]), file=result_file)
    result_file.close()
