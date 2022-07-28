import torch
import tqdm
import time
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from prefetch_generator import BackgroundGenerator
from functools import partial

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from lxmert_model import LXMERT
from category_id_map import category_id_to_lv2id
from utils.util import evaluate


import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

logger = logging.getLogger(__name__)
DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames)
    size = len(dataset)
    val_size = int(size * args.val_ratio)

    train_indices, test_indices = train_test_split(list(range(len(dataset.labels))), test_size=args.val_ratio,
                                                   random_state=2022, stratify=dataset.labels)
    _, val_dataset = torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset,
                                                                                                          test_indices)
    if args.num_workers > 0:
        dataloader_class = partial(DataLoaderX, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoaderX, pin_memory=True, num_workers=0)

    val_sampler = SequentialSampler(val_dataset)

    dataloader = dataloader_class(val_dataset,
                                  batch_size=15,
                                  sampler=val_sampler,
                                  drop_last=False,
                                  pin_memory=False,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)

    # 2. load model
    model = LXMERT(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
            pred_label_id = model(batch[0], batch[1], batch[2], batch[3], inference=True)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open('./test/torch.csv', 'w') as f:
        for pred_label_id, ann in zip(predictions, val_dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')

    # 5. load trt model
    predictions = []
    with load_tensorrt_engine('/opt/ml/wxcode/model.trt.engine') as engine:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            for data in dataloader:
                # batch = {k: v.numpy() for k, v in data.items()}
                outputs_dict = do_inference(data, context, bindings, inputs, outputs, stream)
                predictions.extend(outputs_dict['output_0'])

    # 6. dump trt results
    with open('./test/trt.csv', 'w') as f:
        for pred_label_id, ann in zip(predictions, val_dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, binding_name, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.binding_name = binding_name
        self.shape = shape

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, binding))
        else:
            output_shape = engine.get_binding_shape(binding)
            if len(output_shape) == 3:
                dims = trt.DimsCHW(engine.get_binding_shape(binding))
                output_shape = (dims.c, dims.h, dims.w)
            elif len(output_shape) == 2:
                dims = trt.Dims2(output_shape)
                output_shape = (dims[0], dims[1])
            outputs.append(HostDeviceMem(host_mem, device_mem, binding, output_shape))

    return inputs, outputs, bindings, stream

def do_inference(batch, context, bindings, inputs, outputs, stream):
    assert len(inputs) == 4
    inputs[0].host = np.ascontiguousarray(batch[0], dtype=np.float16)
    inputs[1].host = np.ascontiguousarray(batch[1], dtype=np.int32)
    inputs[2].host = np.ascontiguousarray(batch[2], dtype=np.int32)
    inputs[3].host = np.ascontiguousarray(batch[3], dtype=np.int32)
    # print(batch[0].size())
    # print(batch[1].size())
    # print(batch[2].size())
    # print(batch[3].size())

    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    outputs_dict = {}
    for out in outputs:
        outputs_dict[out.binding_name] = np.reshape(out.host, out.shape)
    return outputs_dict

def load_tensorrt_engine(filename):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(filename, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def evaluate_submission(result_file, ground_truth_file):
    ground_truth = {}
    with open(ground_truth_file, 'r') as f:
        for line in f:
            vid, category_id = line.strip().split(',')
            ground_truth[vid] = category_id_to_lv2id(category_id)

    predictions, labels = [], []
    with open(result_file, 'r') as f:
        for line in f:
            vid, category_id = line.strip().split(',')
            if vid not in ground_truth:
                raise Exception(f'ERROR id {vid} in result.csv')
            predictions.append(category_id_to_lv2id(category_id))
            labels.append(ground_truth[vid])

    if len(predictions) != len(ground_truth):
        raise Exception(f'ERROR: Wrong line numbers')

    return evaluate(predictions, labels)

if __name__ == '__main__':
    inference()
    result_file = 'test/trt.csv'
    ground_truth_file = 'test/torch.csv'

    result = evaluate_submission(result_file, ground_truth_file)
    print(f'mean F1 score is {result["mean_f1"]}')
