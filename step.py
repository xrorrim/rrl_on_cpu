import os
import logging
import numpy as np
import torch
torch.set_num_threads(2)
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict

from rrl.utils import read_csv, DBEncoder
from rrl.models import RRL



def load_model(path, device_id, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['rrl_args']
    rrl = RRL(
        dim_list=saved_args['dim_list'],
        # device_id=device_id,
        is_rank0=True,
        use_not=saved_args['use_not'],
        log_file=log_file,
        distributed=distributed,
        estimated_grad=saved_args['estimated_grad'],
        use_skip=saved_args['use_skip'],
        use_nlaf=saved_args['use_nlaf'],
        alpha=saved_args['alpha'],
        beta=saved_args['beta'],
        gamma=saved_args['gamma'])
    stat_dict = checkpoint['model_state_dict']
    for key in list(stat_dict.keys()):
        # remove 'module.' prefix
        stat_dict[key] = stat_dict.pop(key)
    rrl.net.load_state_dict(checkpoint['model_state_dict'])
    return rrl


def test_model(args,x:torch.Tensor):
    
    rrl = load_model(args.model, args.device_ids[0], log_file=args.test_res, distributed=False)
    resulttensor = rrl.net.bi_forward(x)
    if resulttensor[0][0] > resulttensor[0][1]:
        result = "positive"
    else:
        result = "negative"
    print("result tensor: ",resulttensor)
    print("result: ",result)
    return result


def encode_string(s):
    # 定义字符到数字的映射
    mapping = {'x': [1, 0], 'o': [0, 1], 'b': [0, 0]}
    
    # 分割字符串并编码
    encoded = [mapping[char] for char in s.split(',') if char in mapping]
    
    # 展平列表并转换为 NumPy 数组
    flat_list = [item for sublist in encoded for item in sublist]
    return np.array(flat_list).reshape(1, -1)

def convert_to_tensor(np_array):
    return torch.tensor(np_array, dtype=torch.float32)

if __name__ == '__main__':
    
    input = "o,x,x,o,x,o,o,b,b"

    encoded_vector = encode_string(input)
    tensor = convert_to_tensor(encoded_vector)

    print("input :",input)
    # 打印结果
    print("input embedding:", tensor)
    # print("Shape:", tensor.shape)

    from args import rrl_args


    test_model(rrl_args,x = tensor)
    