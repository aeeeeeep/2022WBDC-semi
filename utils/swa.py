import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from datetime import timedelta
import time
import logging

def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.bin 的路径
    """
    model_lists = []

    for root, dirs, files in os.walk('save/'):
        for _file in files:
            if 'model' in _file:
                model_lists.append(os.path.join(root, _file))

    model_lists = sorted(model_lists)
    return model_lists

def swa(model, model_dir, swa_start=3):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)

    assert 1 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            logging.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join('save/', f'swa')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    logging.info(f'Save swa model in: {swa_model_dir}')

    swa_model_path = os.path.join(swa_model_dir, 'model.bin')

    torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model