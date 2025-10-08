#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2023/10/26 上午9:09
import glob
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stat_data(xlxs_path):
    df = pd.read_excel(xlxs_path, sheet_name='Label')

    category = []
    sub_category = []
    sub_category_cnt = []

    for category_idx in range(1, 15):
        category_cnt = 0
        sub_category_list = []
        for row in range(len(df.index.values)):
            if df.iloc[row, 1] == category_idx:
                category_cnt += 1
                sub_category_list.append(df.iloc[row, 2])

        print(f'category {category_idx}: {category_cnt}, ', end='')

        for sub_category_idx in set(sub_category_list):
            nums = np.sum([c == sub_category_idx for c in sub_category_list])
            print(f'{nums} ({sub_category_idx})', end=' ')
            if nums != 0 and category_cnt != 0:
                category.append(category_idx)
                sub_category.append(sub_category_idx)
                sub_category_cnt.append(nums)
        print()

    return category, sub_category, sub_category_cnt

def split_pretrain():
    pickle_path = '/home/crz/MyositisDataset/Myositis_label.pkl'
    with open(pickle_path, 'rb') as f:
        label_dict = pickle.load(f)

    pickle_path = '/home/crz/MyositisDataset/split_data/06/train_split.pkl'
    with open(pickle_path, 'rb') as f:
        train_dict = pickle.load(f)

    for i in train_dict:
        print(i, int(label_dict.get(i).get('score')))

    print(train_dict)
    print(label_dict)

    idx = [sample for sample in label_dict]

    for i in idx:
        image_list = (glob.glob(os.path.join('/home/crz/MyositisDataset', "square_data", f'{i:04d}*.jpg')))
        print(len(image_list))

        if len(image_list) != 103:
            idx.remove(i)
            print(f'Remove {i}')


    train_idx, test_idx = train_test_split(idx, train_size=0.85)
    print('--- ', len(train_idx))

    pickle_path = '/home/crz/MyositisDataset/split_data/pretrain/train_split.pkl'
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump(train_idx, f)

    pickle_path = '/home/crz/MyositisDataset/split_data/pretrain/test_split.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(test_idx, f)

    data_path = '/home/crz/MyositisDataset/split_data/pretrain/data'
    with open(data_path, 'w') as f:
        for i in idx:
            f.write(str(i) + '\n')


def train_test_split_ous(data_list, test_size, shuffle=None):
    total_num = len(data_list)
    test_num = total_num * test_size
    intervel = total_num / test_num
    test_split = [data_list[int(intervel * i)] for i in range(int(test_num))]
    train_split = list(set(data_list) - set(test_split))

    return train_split, test_split


def split_data(xlxs_path, test_size=0.3):
    df = pd.read_excel(xlxs_path, sheet_name='Label')

    category = []
    sub_category = []
    sub_category_cnt = []

    for category_idx in range(1, 15):
        category_cnt = 0
        sub_category_list = []
        num_list = []
        for row in range(len(df.index.values)):
            if df.iloc[row, 1] == category_idx:
                category_cnt += 1
                sub_category_list.append(df.iloc[row, 2])
                num_list.append(df.iloc[row, 0])

        print(f'category {category_idx}: {category_cnt}, ', end='')


        train_splits = []
        test_splits = []
        for sub_category_idx in set(sub_category_list):

            nums = np.sum([c == sub_category_idx for c in sub_category_list])

            print(f'{nums} ({sub_category_idx})', end=' ')
            if nums != 0 and category_cnt != 0:
                category.append(category_idx)
                sub_category.append(sub_category_idx)
                sub_category_cnt.append(nums)

            # split data
            sub_category_idx_list = []
            for pos, c in enumerate(sub_category_list):
                if c == sub_category_idx:
                    sub_category_idx_list.append(num_list[pos])

            if len(sub_category_idx_list) >= 5:
                train_split, test_split = train_test_split_ous(
                    sub_category_idx_list, test_size=test_size, shuffle=True)
                print()
                print(sub_category_idx_list)
                print(test_split)
            else:
                train_split, test_split = sub_category_idx_list, []
            train_splits = train_splits + train_split
            test_splits = test_splits + test_split

        print()

        # save splits
        if category_cnt != 0:
            split_name = 'split_data_cl-ous'

            os.makedirs(f'../data/jdm/{split_name}/{category_idx:02d}', exist_ok=True)

            pickle_path = f'../data/jdm/{split_name}/{category_idx:02d}/train_split.pkl'
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(sorted(train_splits), f)

            pickle_path = f'../data/jdm/{split_name}/{category_idx:02d}/test_split.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(sorted(test_splits), f)

            data_path = f'../data/jdm/{split_name}/{category_idx:02d}/data'
            with open(data_path, 'w') as f:
                for i in sorted(num_list):
                    f.write(str(i) + '\n')

            print(f'save data to {data_path}')

if __name__ == '__main__':
    xlxs_path = '/home/crz/MyositisDataset/Myositis_Label.xlsx'
    split_data(xlxs_path)

    pass