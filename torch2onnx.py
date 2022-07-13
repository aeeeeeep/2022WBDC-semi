import time
import onnx
import torch
import torchvision
import onnxruntime
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
import numpy as np
from tqdm import tqdm
import onnx
import onnxruntime
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
import time
from albef_model import ALBEF

args = parse_args()

args = parse_args()
# 1. load data
dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True)
sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset,
                        batch_size=args.test_batch_size,
                        sampler=sampler,
                        drop_last=False,
                        pin_memory=True,
                        num_workers=args.num_workers,
                        prefetch_factor=args.prefetch)

# 1. 定义模型
model = ALBEF(args).cuda()

# 2.定义输入&输出
input_names = ['input']
output_names = ['output']

x = iter(dataloader)
x = next(x)
a = torch.ones([32, 1]).to(torch.Tensor()).to("cuda")

# 3.pt转onnx
onnx_file = "./save/model.onnx"
torch.onnx.export(model, (x['frame_input'].to(torch.Tensor()).to("cuda"),x['frame_mask'].to(torch.Tensor()).to("cuda"),x['text_input'].to(torch.Tensor()).to("cuda"),x['text_mask'].to(torch.Tensor()).to("cuda"), a),'model.onnx',export_params=True,verbose=True,input_names=input_names,output_names=output_names)


# 4.检查onnx计算图
net = onnx.load("./model.onnx")
onnx.checker.check_model(net)           # 检查文件模型是否正确

# 5.优化前后对比&验证
# 优化前
model.eval()
with torch.no_grad():
    output1 = model(x['frame_input'].to(torch.Tensor()).to("cuda"),x['frame_mask'].to(torch.Tensor()).to("cuda"),x['text_input'].to(torch.Tensor()).to("cuda"),x['text_mask'].to(torch.Tensor()).to("cuda"), a)

# 优化后
inputs = (x['frame_input'].to(torch.Tensor()).to("cuda"),x['frame_mask'].to(torch.Tensor()).to("cuda"),x['text_input'].to(torch.Tensor()).to("cuda"),x['text_mask'].to(torch.Tensor()).to("cuda"), a)
session = onnxruntime.InferenceSession("./model.onnx")
session.get_modelmeta()
output2 = session.run(['output'], {"input": inputs.cpu().numpy()})
print("{}vs{}".format(output1.mean(), output2[0].mean()))