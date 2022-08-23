import os
import random

import torch
import numpy as np
from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping
import wandb
import optuna
import joblib
import datetime

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1, K=1, n_items=0):
    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs * K)).to(device)
    return feed_dict


def opt_objective(trial, args, train_cf, user_dict, n_params, norm_mat, deg):
    valid_res_list = []

    args.dim = trial.suggest_int('dim', 64, 512)
    args.l2 = trial.suggest_float('l2', 0, 0.1)
    args.context_hops = trial.suggest_int('context_hops', 1, 6)

    for run in range(args.runs):
        # args.run = i + 1
        for seed in range(1):
            torch.cuda.manual_seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(args)
            valid_best_result = main(args, run, train_cf, user_dict, n_params, norm_mat, deg)
            valid_res_list.append(valid_best_result)
    # print(valid_res_list)
    # print("args: ", args)
    # print(f'Obj valid avg recall: {np.mean(valid_res_list)}')
    return np.mean(valid_res_list)


def main(args, run, train_cf, user_dict, n_params, norm_mat, deg):
    wandb.login(key="ed022f13b9f1b9e155450b10e06c563b40452b07")
    wandb.init(
        name='100-' + args.gnn + '-' + args.dataset + '-hops-' + str(args.context_hops) + '-t_u-' +
             str(args.t_u) + '-t_i-' + str(args.t_i) + '-dim-' + str(args.dim) + '-l2-' + str(args.l2),
        project="TKDE-ApeGNN",
        entity="keg-aminer-rec")
    """define model"""
    from ApeGNN import ApeGNN
    from LightGCN import LightGCN
    from NGCF import NGCF
    from ApeGNN_Per import ApeGNN_Per
    from ApeGNN_Per_Deg import ApeGNN_Per_Deg
    if args.gnn == 'ApeGNN':
        model = ApeGNN(n_params, args, norm_mat).to(device)
    elif args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    elif args.gnn == 'ngcf':
        model = NGCF(n_params, args, norm_mat).to(device)
    elif args.gnn == 'ApeGNN_Per':
        model = ApeGNN_Per(n_params, args, norm_mat).to(device)
    elif args.gnn == 'ApeGNN_Per_Deg':
        model = ApeGNN_Per_Deg(n_params, args, norm_mat, deg).to(device)
    """define optimizer"""
    optimizer = torch.optim.Adam([{'params': model.parameters(),
                                   'lr': args.lr}])
    n_items = n_params['n_items']
    cur_best_pre_0 = 0
    stopping_step = 0
    # should_stop = False
    best_epoch = 0
    print("start training ...")

    hyper = {"dim": args.dim, "l2": args.l2, "hops": args.context_hops}
    print("Start hyper parameters: ", hyper)
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  args.n_negs,
                                  args.K,
                                  n_items)
            batch_loss, _, _ = model(batch)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size
        train_e_t = time()
        print('loss:', round(loss.item(), 2), "time: ", round(train_e_t - train_s_t, 2), 's')

        if epoch % args.step == 0:
            """testing"""

            train_res = PrettyTable()
            train_res.field_names = ["Phase", "Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg",
                                     "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                ["Test", epoch, round(train_e_t - train_s_t, 2), round(test_e_t - test_s_t, 2), round(loss.item(), 2),
                 test_ret['recall'],
                 test_ret['ndcg'],
                 test_ret['hit_ratio'],
                 test_ret['precision']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    ["Valid", epoch, round(train_e_t - train_s_t, 2), round(test_e_t - test_s_t, 2), round(loss.item(), 2),
                     valid_ret['recall'],
                     valid_ret['ndcg'],
                     valid_ret['hit_ratio'],
                     valid_ret['precision']])
            print(train_res)
            if len(eval(args.Ks)) == 1:
                es_index_20 = 0
            elif len(eval(args.Ks)) == 2:
                es_index_20 = 0
                es_index_50 = 1
            else:
                es_index_20 = 2
                es_index_50 = 3
            wandb_log = {}
            wandb_log["Epoch"] = epoch
            wandb_log["TrainLoss"] = loss / args.batch_size
            wandb_log["Time"] = test_e_t - train_s_t
            # wandb_log["dataset"] = args.dataset
            # wandb_log["recall@5"] = valid_ret['recall'][0]
            # wandb_log["recall@10"] = valid_ret['recall'][1]
            wandb_log["recall_val@20"] = valid_ret['recall'][es_index_20]
            wandb_log["recall_test@20"] = test_ret['recall'][es_index_20]
            wandb_log["recall_val@50"] = valid_ret['recall'][es_index_50]
            wandb_log["recall_test@50"] = test_ret['recall'][es_index_50]
            # wandb_log["recall@50"] = valid_ret['recall'][3]
            # wandb_log["ndcg@5"] = valid_ret['ndcg'][0]
            # wandb_log["ndcg@10"] = valid_ret['ndcg'][1]
            wandb_log["ndcg_val@20"] = valid_ret['ndcg'][es_index_20]
            wandb_log["ndcg_test@20"] = test_ret['ndcg'][es_index_20]
            wandb_log["ndcg_val@50"] = valid_ret['ndcg'][es_index_50]
            wandb_log["ndcg_test@50"] = test_ret['ndcg'][es_index_50]
            # wandb_log["ndcg@50"] = valid_ret['ndcg'][3]
            # wandb_log["hit_ratio@5"] = valid_ret['hit_ratio'][0]
            # wandb_log["hit_ratio@10"] = valid_ret['hit_ratio'][1]
            wandb_log["hit_ratio_val@20"] = valid_ret['hit_ratio'][es_index_20]
            wandb_log["hit_ratio_test@20"] = test_ret['hit_ratio'][es_index_20]
            wandb_log["hit_ratio_val@50"] = valid_ret['hit_ratio'][es_index_50]
            wandb_log["hit_ratio_test@50"] = test_ret['hit_ratio'][es_index_50]
            # wandb_log["hit_ratio@50"] = valid_ret['hit_ratio'][3]
            # wandb_log["precision@5"] = valid_ret['precision'][0]
            # wandb_log["precision@10"] = valid_ret['precision'][1]
            wandb_log["precision_val@20"] = valid_ret['precision'][es_index_20]
            wandb_log["precision_test@20"] = test_ret['precision'][es_index_20]
            wandb_log["precision_val@50"] = valid_ret['precision'][es_index_50]
            wandb_log["precision_test@50"] = test_ret['precision'][es_index_50]
            # wandb_log["precision@50"] = valid_ret['precision'][3]
            # wandb_log["hops"] = args.context_hops
            # wandb_log["l2"] = args.l2
            # wandb_log['dim'] = args.dim

            wandb.log(wandb_log)
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.

            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][es_index_20], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if valid_ret['recall'][es_index_20] == cur_best_pre_0:
                best_epoch = epoch
            if should_stop:
                break

            """save weight"""
            if valid_ret['recall'][es_index_20] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + f'{args.dataset}_{args.dim}_{args.context_hops}_{args.l2}_' + args.gnn + '.ckpt')
                best_epoch = epoch
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f, best_epoch at %d' % (epoch, cur_best_pre_0, best_epoch))
    print("Seed:", run)
    print("End hyper parameters: ", hyper)
    print(f"Best valid_ret['recall']: ", cur_best_pre_0)
    return cur_best_pre_0


if __name__ == '__main__':
    """read args"""
    global args, device
    args = parse_args()
    s = datetime.datetime.now()
    print("time of start: ", s)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # torch.cuda.set_device(args.gpu_id)
    # device = args.gpu_id
    """build dataset"""
    train_cf, user_dict, n_params, norm_mat, deg = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    trials = 4
    search_space = {'dim': [64, 128, 256, 512], 'context_hops': [1, 2, 3, 4], 'l2': [1e-4]}
    if args.gnn in ['lightgcn']:
        if args.dataset in ['ml-1m']:
            search_space = {'dim': [64], 'context_hops': [1, 2, 3, 4], 'l2': [1e-4]}
            trials = 4
    elif args.gnn in ['ngcf']:
        if args.dataset in ['ml-1m']:
            search_space = {'dim': [64], 'context_hops': [1, 2, 3, 4], 'l2': [1e-5]}
            trials = 4
    else:
        if args.dataset in ['aminer', 'ml-1m', 'gowalla', 'yelp2018', 'amazon', 'ali']:
            if args.gs_l2 != 1:
                search_space = {'dim': [64, 128, 256, 512], 'context_hops': [1, 2, 3, 4], 'l2': [1e-4]}
                trials = 16
            else:
                search_space = {'dim': [64, 128, 256, 512], 'context_hops': [1, 2, 3, 4], 'l2': [1e-2, 1e-3, 1e-5]}
                trials = 48
    print("search_space: ", search_space)
    print("trials: ", trials)
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: opt_objective(trial, args, train_cf, user_dict, n_params, norm_mat, deg), n_trials=trials)
    joblib.dump(study,
                f'{args.dataset}_{args.dim}_{args.context_hops}_{args.l2}_study_' + args.gnn + '.pkl')
    # run_best(args, study)
    e = datetime.datetime.now()
    print(study.best_trial.params)
    print("time of end: ", e)
