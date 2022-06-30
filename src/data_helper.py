import json
import random
import zipfile
from io import BytesIO
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from category_id_map import category_id_to_lv2id

def create_dataloaders(args, val_idx=0):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    val_size = int(len(dataset) * args.val_ratio)
    fold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    fold[0], fold[1], fold[2], fold[3], fold[4], fold[5], fold[6], fold[7], fold[8], fold[9] = \
            torch.utils.data.random_split(dataset, [val_size, val_size, val_size, val_size, val_size, val_size, val_size, val_size, val_size, val_size], \
                                      generator=torch.Generator().manual_seed(args.seed))

    if val_idx == 1:
        train_dataset = fold[0] + fold[1] + fold[2] + fold[3] + fold[4] + fold[5] + fold[6] + fold[7] + fold[8]
        val_dataset = fold[9]
    elif val_idx == 2:
        train_dataset = fold[0] + fold[1] + fold[2] + fold[3] + fold[4] + fold[5] + fold[6] + fold[7] + fold[9]
        val_dataset = fold[8]
    elif val_idx == 3:
        train_dataset = fold[0] + fold[1] + fold[2] + fold[3] + fold[4] + fold[5] + fold[6] + fold[9] + fold[8]
        val_dataset = fold[7]
    elif val_idx == 4:
        train_dataset = fold[0] + fold[1] + fold[2] + fold[3] + fold[4] + fold[5] + fold[9] + fold[7] + fold[8]
        val_dataset = fold[6]
    elif val_idx == 5:
        train_dataset = fold[0] + fold[1] + fold[2] + fold[3] + fold[4] + fold[9] + fold[6] + fold[7] + fold[8]
        val_dataset = fold[5]
    elif val_idx == 6:
        train_dataset = fold[0] + fold[1] + fold[2] + fold[3] + fold[9] + fold[5] + fold[6] + fold[7] + fold[8]
        val_dataset = fold[4]
    elif val_idx == 7:
        train_dataset = fold[0] + fold[1] + fold[2] + fold[9] + fold[4] + fold[5] + fold[6] + fold[7] + fold[8]
        val_dataset = fold[3]
    elif val_idx == 8:
        train_dataset = fold[0] + fold[1] + fold[9] + fold[3] + fold[4] + fold[5] + fold[6] + fold[7] + fold[8]
        val_dataset = fold[2]
    elif val_idx == 9:
        train_dataset = fold[0] + fold[9] + fold[2] + fold[3] + fold[4] + fold[5] + fold[6] + fold[7] + fold[8]
        val_dataset = fold[1]
    else:
        train_dataset = fold[9] + fold[1] + fold[2] + fold[3] + fold[4] + fold[5] + fold[6] + fold[7] + fold[8]
        val_dataset = fold[0]

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache,
                                                       add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            if self.test_mode:
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, title: str, ocr_text: str, asr_text: str) -> tuple:
        max_len = 128
        if len(title) >= max_len:
            title = title[:(int(max_len / 2))] + title[-(int(max_len / 2)):]
        if len(ocr_text) >= max_len:
            ocr_text = ocr_text[:(int(max_len / 2))] + ocr_text[-(int(max_len / 2)):]
        if len(asr_text) >= max_len:
            asr_text = asr_text[:(int(max_len / 2))] + asr_text[-(int(max_len / 2)):]

        encoded_title = self.tokenizer(title, max_length=max_len, padding='max_length', truncation=True)
        encoded_ocr = self.tokenizer(ocr_text, max_length=max_len, padding='max_length', truncation=True)
        encoded_asr = self.tokenizer(asr_text, max_length=max_len, padding='max_length', truncation=True)

        text_input_ids = torch.LongTensor(
            [self.tokenizer.cls_token_id] + encoded_title['input_ids'] + [self.tokenizer.sep_token_id]
            + encoded_ocr['input_ids'] + [self.tokenizer.sep_token_id] + encoded_asr['input_ids']
            + [self.tokenizer.sep_token_id]
        )
        text_mask = torch.LongTensor(
            [1, ] + encoded_title['attention_mask'] + [1, ] + encoded_ocr['attention_mask'] + [1, ]
            + encoded_asr['attention_mask'] + [1, ]
        )
        text_token_type_ids = torch.zeros_like(text_input_ids)
        return text_input_ids, text_mask, text_token_type_ids

    def tokenize_frame(self, idx: int) -> tuple:
        frame_input, frame_mask = self.get_visual_feats(idx)
        frame_token_type_ids = torch.ones_like(frame_mask)
        return frame_input, frame_mask, frame_token_type_ids

    def __getitem__(self, idx: int) -> dict:
        frame_input, frame_mask, frame_token_type_ids = self.tokenize_frame(idx)

        title, asr = self.anns[idx]['title'], self.anns[idx]['asr']
        ocr = sorted(self.anns[idx]['ocr'], key=lambda x: x['time'])
        ocr = ','.join([t['text'] for t in ocr])
        text_input, text_mask, text_token_type_ids = self.tokenize_text(title, ocr, asr)

        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            frame_token_type_ids=frame_token_type_ids,
            text_input=text_input,
            text_mask=text_mask,
            text_token_type_ids=text_token_type_ids
        )

        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
