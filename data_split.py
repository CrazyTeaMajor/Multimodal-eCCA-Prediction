import os
import random

import pandas as pd
import numpy as np
import random as rd
import dataset
from torch.utils.data.dataloader import DataLoader

task_list = ["PD_L1", "VEGF"]
data_list = ["DWI", "T1WI", "T2WI"]


def count_list(file_path):
    cnt = 0
    file_list = os.listdir(file_path)
    if len(file_list) == 1 and len(file_list[0].split('.')) == 1:
        file_path = "{}/{}".format(file_path, file_list[0])
        file_list = os.listdir(file_path)
    for file_name in file_list:
        if file_name.split('.')[-1] == 'dcm':
            cnt += 1
    return cnt


org_path = "./DeepLearning_Data"
for task in task_list:
    now_path = "{}/{}".format(org_path, task)
    for data in data_list:
        cnt0 = 0
        cnt1 = 0
        cnt_path = "{}/Label(-)/Original_Images".format(now_path)
        p_list = os.listdir(cnt_path)
        for p_name in p_list:
            cnt0 += count_list("{}/{}/{}".format(cnt_path, p_name, data))
        cnt_path = "{}/Label(+)/Original_Images".format(now_path)
        p_list = os.listdir(cnt_path)
        for p_name in p_list:
            cnt1 += count_list("{}/{}/{}".format(cnt_path, p_name, data))
        print("{}-{}: 0-{}, 1-{} all-{}".format(task, data, cnt0, cnt1, cnt0+cnt1))




# task_type = "VEGF"
# task_type = "PDL1"
# test_data = dataset.CCA_dataset('{}'.format(task_type), "./{}_test.txt".format(task_type))
# train_data = dataset.CCA_dataset('{}'.format(task_type), "./{}_train.txt".format(task_type))
# test_loader = DataLoader(test_data, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
# train_loader = DataLoader(train_data, batch_size=1, num_workers=0, pin_memory=False, shuffle=False)
# print(task_type, "DWI")
#
# cnt0 = 0
# cnt1 = 0
# for f1, f2, DWI, TW1, TW2, label in train_loader:
#     if label[0] == 0:
#         cnt0 += len(DWI)
#     if label[0] == 1:
#         cnt1 += len(DWI)
#
# print("train:", cnt0, cnt1)
# cnt0_all = cnt0
# cnt1_all = cnt1
# cnt0 = 0
# cnt1 = 0
#
# for f1, f2, DWI, TW1, TW2, label in test_loader:
#     if label[0] == 0:
#         cnt0 += len(DWI)
#     if label[0] == 1:
#         cnt1 += len(DWI)
#
# print("test:", cnt0, cnt1)
# cnt0_all += cnt0
# cnt1_all += cnt1
# print("all:", cnt0_all, cnt1_all)

# now_df = open("./PDL1_train.txt", 'r')
#
# id_list = []
# for i in now_df:
#     i = i.strip()
#     id_list.append(i)
#
# random.shuffle(id_list)
#
# for fold in range(4):
#     train_txt = open("./PDL1_train_fold{}.txt".format(fold), "w")
#     val_txt = open("./PDL1_val_fold{}.txt".format(fold), "w")
#     for i in range(76):
#         if fold * 19 <= i < (fold + 1) * 19:
#             val_txt.write("{}\n".format(id_list[i]))
#         else:
#             train_txt.write("{}\n".format(id_list[i]))
#     train_txt.close()
#     val_txt.close()


# df = pd.read_excel('./Clinical_Features/Clinical_Data_Processed.xlsx')
#
# pdl1_0 = []
# pdl1_1 = []
#
# vegf_0 = []
# vegf_1 = []
#
# for index, row in df.iterrows():
#     id = row['patients']
#     pdl1 = int(row['pdl1'])
#     vegf = int(row['vegf'])
#     if pdl1 == 1:
#         pdl1_1.append(id)
#     else:
#         pdl1_0.append(id)
#     if vegf == 1:
#         vegf_1.append(id)
#     else:
#         vegf_0.append(id)
#
# rd.seed = 42
# rd.shuffle(pdl1_0)
# rd.shuffle(pdl1_1)
# rd.shuffle(vegf_0)
# rd.shuffle(vegf_1)
#
# print(len(pdl1_0), len(pdl1_1))
# print(len(vegf_0), len(vegf_1))
#
# split_data_pdl1_test = open("PDL1_test.txt", 'w', newline='')
# split_data_pdl1_train = open("PDL1_train.txt", 'w', newline='')
# split_data_vegf_test = open("VEGF_test.txt", 'w', newline='')
# split_data_vegf_train = open("VEGF_train.txt", 'w', newline='')
#
# for i in range(10):
#     split_data_pdl1_test.write('{}\n'.format(pdl1_1[i]))
#     split_data_pdl1_test.write('{}\n'.format(pdl1_0[i]))
#     split_data_vegf_test.write('{}\n'.format(vegf_1[i]))
#     split_data_vegf_test.write('{}\n'.format(vegf_0[i]))
#
# for i in range(10, len(pdl1_0)):
#     split_data_pdl1_train.write('{}\n'.format(pdl1_0[i]))
# for i in range(10, len(pdl1_1)):
#     split_data_pdl1_train.write('{}\n'.format(pdl1_1[i]))
# for i in range(10, len(vegf_0)):
#     split_data_vegf_train.write('{}\n'.format(vegf_0[i]))
# for i in range(10, len(vegf_1)):
#     split_data_vegf_train.write('{}\n'.format(vegf_1[i]))
#
# split_data_pdl1_train.close()
# split_data_vegf_train.close()
# split_data_vegf_test.close()
# split_data_pdl1_test.close()
