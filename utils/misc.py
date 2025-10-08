# -*- coding: utf-8 -*-
# @Time: 2023/6/20 20:37
import argparse
import os
import random
import shutil

import numpy as np
import pickle
import torch


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class cmdAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + '_non_default', True)


def copy_dir(src, dst):
    if os.path.exists(dst) and os.path.isdir(dst):
        shutil.rmtree(dst)

    if os.path.exists(src):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)


def init_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def mask_classes(outputs, dataset, k):
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def distributed_concat(data):
    if torch.distributed.is_initialized() == False:
        return data

    if isinstance(data, dict):
        # 如果是字典，递归处理每个键值对
        concatenated_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # 对张量进行分布式拼接
                output_tensors = [
                    value.clone().contiguous()
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(output_tensors, value.contiguous())
                concatenated_data[key] = torch.cat(output_tensors, dim=0)
            else:
                # 如果值不是张量，直接保留原值
                concatenated_data[key] = value
        return concatenated_data
    else:
        # 如果不是字典，假设是张量，直接处理
        output_tensors = [
            data.clone().contiguous() for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output_tensors, data.contiguous())
        concat_tensor = torch.cat(output_tensors, dim=0)
        return concat_tensor


def read_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)

    return pickle_data
