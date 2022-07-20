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
from lxmert_model_inference import LXMERT
from onnxmltools.utils import float16_converter
from onnx import load_model, save_model

# Step 1, build model
args = parse_args()
model = LXMERT(args)
chkpt = torch.load('save/v4/model_epoch_2_mean_f1_0.6854.bin', map_location='cpu')['model_state_dict']
model.load_state_dict(chkpt,strict=False)
model.eval()

# Step 2, export model to onnx （这里使用固定尺寸，动态尺寸容易出问题）
frame_input = torch.zeros(10, 12, 3, 224, 224).to(torch.float32)
frame_mask = torch.zeros(10, 12).to(torch.int32)
title_input = torch.zeros(10, 388).to(torch.int32)
title_mask = torch.zeros(10, 388).to(torch.int32)
torch.onnx.export(model, (frame_input,frame_mask,title_input,title_mask), './save/model.onnx', verbose=False, opset_version=12,
                  input_names=["input_0"],
                  output_names=["output_0"],
                  do_constant_folding=False,
                  use_external_data_format=False,
                  enable_onnx_checker=True)