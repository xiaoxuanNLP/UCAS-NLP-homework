import torch
import json
import numpy as np
from .config import *
from os.path import dirname, join
from os import remove

_root_dir = dirname(__file__)
MODEL_PARAM = join(_root_dir, 'param')

def subsequent_mask(size):
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # 生成一个三角形矩阵

    return torch.from_numpy(subsequent_mask) == 0

def sequence_mask(ids,max_len):
    mask = []
    for i in range(max_len):
        if i < len(ids):
            mask.append(1)
        else:
            mask.append(0)

    return mask

def save_training_state(epoch_i,
                        loss,
                        model_state,
                        save_all,
                        is_best_loss,
                        prev_model_file_path):
    model_save_name = ModelSaveName.format(
        epoch_i,loss
    )

    model_param_path = join(MODEL_PARAM, model_save_name)
    print("model_param_path = ", model_param_path)

    if save_all:
        torch.save(model_state, model_param_path)
        return None
    elif is_best_loss:
        if prev_model_file_path is not None:
            remove(prev_model_file_path)

        torch.save(model_state, model_param_path)
        return model_param_path
    else:
        return prev_model_file_path
