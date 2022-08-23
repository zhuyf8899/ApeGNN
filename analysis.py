# -*- !/usr/bin/env python -*-
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     analysis
   Description :
   Author :       zhangdan
   date：          2022/8/22
-------------------------------------------------
   Change Activity:
                   2022/8/22:
-------------------------------------------------
"""
__author__ = 'zhangdan'

# analyze relation between t and degree
import torch
from ApeGNN_Per import ApeGNN_Per
from utils.parser import parse_args
from utils.data_loader import load_data
from numpy import *
import matplotlib.pyplot as plt
import math

args = parse_args()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
_, _, n_params, norm_mat = load_data(args)
model = ApeGNN_Per(n_params, args, norm_mat).to(device)
aminer_ckpt_model = torch.load('./weights/aminer_64_2_1e-05_ApeGNN_Per.ckpt')
# aminer_ckpt_model = torch.load('./weights/aminer_256_1_1e-05_ApeGNN_Per.ckpt')
# aminer_ckpt_model = torch.load('./weights/gowalla_256_4_0.001_ApeGNN_Per.ckpt')
# params = model.load_state_dict(aminer_ckpt_model)
print("Model's state_dict: ")
u_ts_0, u_ts_1, u_ts_2, u_ts_3, u_ts_4 = dict(), dict(), dict(), dict(), dict()
idx = 0
for param in aminer_ckpt_model:
    print(param, "\t", aminer_ckpt_model[param].size(), "\t", aminer_ckpt_model[param])
# for param in model.state_dict():
#     print(param, "\t", model.state_dict()[param].size(), "\t", model.state_dict()[param])
for u_t in aminer_ckpt_model['gcn.user_t']:
    # print(u_t.cpu().numpy(), type(u_t))
    t = u_t.cpu().numpy()[0]
    u_ts_0[idx] = math.exp(-t) * (pow(t, 0) / math.factorial(0))  # t / (exp(t) * math.factorial(0))
    u_ts_1[idx] = math.exp(-t) * (pow(t, 1) / math.factorial(1))  # t / (exp(t) * math.factorial(1))
    u_ts_2[idx] = math.exp(-t) * (pow(t, 2) / math.factorial(2))  # t / (exp(t) * math.factorial(2))
    u_ts_3[idx] = math.exp(-t) * (pow(t, 3) / math.factorial(3))  # t / (exp(t) * math.factorial(3))
    u_ts_4[idx] = math.exp(-t) * (pow(t, 4) / math.factorial(4))  # t / (exp(t) * math.factorial(4))
    idx += 1

lines = open('./data/' + args.dataset + '/train.txt', "r").readlines()
u_degrees = dict()
for l in lines:
    tmps = l.strip()
    inters = [int(i) for i in tmps.split(" ")]
    u_id, pos_ids = inters[0], inters[1:]
    u_degrees[u_id] = len(set(pos_ids))
    # print(u_id, len(set(pos_ids)), u_ts[u_id])

degree_ut_0, degree_ut_1, degree_ut_2, degree_ut_3, degree_ut_4 = dict(), dict(), dict(), dict(), dict()
for uid, degree in u_degrees.items():
    if degree not in degree_ut_1.keys():
        degree_ut_0[degree] = []
        degree_ut_0[degree].append(u_ts_0[uid])

        degree_ut_1[degree] = []
        degree_ut_1[degree].append(u_ts_1[uid])

        degree_ut_2[degree] = []
        degree_ut_2[degree].append(u_ts_2[uid])

        degree_ut_3[degree] = []
        degree_ut_3[degree].append(u_ts_3[uid])

        degree_ut_4[degree] = []
        degree_ut_4[degree].append(u_ts_4[uid])
    else:
        # ts = degree_ut[degree]
        degree_ut_0[degree].append(u_ts_0[uid])
        degree_ut_1[degree].append(u_ts_1[uid])
        degree_ut_2[degree].append(u_ts_2[uid])
        degree_ut_3[degree].append(u_ts_3[uid])
        degree_ut_4[degree].append(u_ts_4[uid])

d_res_0 = sorted(degree_ut_0.items(), key=lambda x: x[0], reverse=False)
d_res_1 = sorted(degree_ut_1.items(), key=lambda x: x[0], reverse=False)
d_res_2 = sorted(degree_ut_2.items(), key=lambda x: x[0], reverse=False)
d_res_3 = sorted(degree_ut_3.items(), key=lambda x: x[0], reverse=False)
d_res_4 = sorted(degree_ut_4.items(), key=lambda x: x[0], reverse=False)

x_data_0, x_data_1, x_data_2, x_data_3, x_data_4 = [], [], [], [], []
y_data_0, y_data_1, y_data_2, y_data_3, y_data_4 = [], [], [], [], []
print("layer0:")
for index0 in d_res_0:
    print(index0[0], index0[1], mean(index0[1]))
    x_data_0.append(index0[0])
    y_data_0.append(mean(index0[1]))

print("layer1:")
for index1 in d_res_1:
    print(index1[0], index1[1], mean(index1[1]))
    x_data_1.append(index1[0])
    y_data_1.append(mean(index1[1]))

print("layer2:")
for index2 in d_res_2:
    print(index2[0], index2[1], mean(index2[1]))
    x_data_2.append(index2[0])
    y_data_2.append(mean(index2[1]))

print("layer3:")
for index3 in d_res_3:
    print(index3[0], index3[1], mean(index3[1]))
    x_data_3.append(index3[0])
    y_data_3.append(mean(index3[1]))

print("layer4:")
for index4 in d_res_4:
    print(index4[0], index4[1], mean(index4[1]))
    x_data_4.append(index4[0])
    y_data_4.append(mean(index4[1]))

plt.plot(x_data_0, y_data_0, alpha=0.5, linewidth=1, label='layer-0')
plt.plot(x_data_1, y_data_1, alpha=0.5, linewidth=1, label='layer-1')
plt.plot(x_data_2, y_data_2, alpha=0.5, linewidth=1, label='layer-2')
plt.plot(x_data_3, y_data_3, alpha=0.5, linewidth=1, label='layer-3')
plt.plot(x_data_4, y_data_4, alpha=0.5, linewidth=1, label='layer-4')

plt.legend(['layer-0', 'layer-1', 'layer-2', 'layer-3', 'layer-4'])
plt.xlabel('degree')
plt.ylabel('mean $u_t$')

plt.savefig('./Vis/aminer_mean_64_2.pdf')
# plt.savefig('./Vis/aminer_mean_256_1.pdf')

# plt.savefig('./Vis/gowalla_mean_256_4.pdf')
