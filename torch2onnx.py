import time
import onnx
import torch
import torchvision
import onnxruntime
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
import numpy as np
from tqdm import tqdm
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
import time
from albef_model import ALBEF


# Step 1, build model
args = parse_args()
model = ALBEF(args)
chkpt = torch.load('save/v2/model_epoch_2_mean_f1_0.6594.bin', map_location='cpu')['model_state_dict']
model.load_state_dict(chkpt)
model.eval()

# Step 2, export model to onnx （这里使用固定尺寸，动态尺寸容易出问题）
frame_input = torch.zeros(10, 8, 3, 224, 224).to(torch.float32)
frame_mask = torch.zeros(10, 8).to(torch.long)
title_input = torch.zeros(10, 32).to(torch.long)
title_mask = torch.zeros(10, 32).to(torch.long)
torch.onnx.export(model, [frame_input,frame_mask,title_input,title_mask], './save/model.onnx', verbose=False, opset_version=12,
                  input_names=["input_0"],
                  output_names=["output_0"],
                  do_constant_folding=True)
