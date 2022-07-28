import os
import zipfile
import torch
import random
from PIL import Image
from io import BytesIO
import tqdm
import numpy as np
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from utils.swin import swin_tiny

args = parse_args()
# 1. load data
dataset = MultiModalDataset(args, args.pretrain_annotation, args.pretrain_zip_frames, test_mode=True)
anns = dataset.dataset.anns
sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset,
                        batch_size=args.test_batch_size,
                        sampler=sampler,
                        drop_last=False,
                        pin_memory=False,
                        num_workers=args.num_workers,
                        prefetch_factor=args.prefetch)

model = swin_tiny(args.swin_pretrained_path)
if torch.cuda.is_available():
    model = torch.nn.parallel.DataParallel(model.cuda())
model.eval()

with torch.no_grad():
    with tqdm.tqdm(total=len(dataloader)) as _tqdm:
        for batch_id, batch in enumerate(dataloader):
            frame_feas = model(batch[0])
            for i in range(args.batch_size):
                frame_np = frame_feas[i,:].cpu().numpy()
                np.save("/home/tione/notebook/data/zip_frames_npy/", frame_np)
            _tqdm.update(1)

        # 4. dump results
with open(args.test_output_csv, 'w') as f:
    for pred_label_id, ann in zip(frames, dataset.anns):
        video_id = ann['id']
        category_id = lv2id_to_category_id(pred_label_id)
        f.write(f'{video_id},{category_id}\n')