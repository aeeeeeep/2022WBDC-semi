import os
import json
import re
import zipfile
import random
import zipfile
import torch
import numpy as np
from PIL import Image
# import cv2
from io import BytesIO
from functools import partial
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from prefetch_generator import BackgroundGenerator
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
from sklearn.model_selection import train_test_split
from collections import Counter

from category_id_map import category_id_to_lv2id

import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def create_dataloaders(args, pretrain=False):
    if pretrain:
        dataset = MultiModalDataset(args, args.pretrain_annotation, args.pretrain_zip_frames, test_mode=True)
        size = len(dataset)
        val_size = 5000
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [size - val_size, val_size])
    else:
        dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames)
        size = len(dataset)
        val_size = int(size * args.val_ratio)
        # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
        #                                                            generator=torch.Generator().manual_seed(args.seed))

        train_indices, test_indices = train_test_split(list(range(len(dataset.labels))), test_size=args.val_ratio,
                                                       random_state=2022, stratify=dataset.labels)
        train_dataset, val_dataset = torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, test_indices)
        resample(train_dataset)

    if args.num_workers > 0:
        dataloader_class = partial(DataLoaderX, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoaderX, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    # kwargs = {"num_workers": args.num_workers, "pin_memory": True}

    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=False,
                                        # **kwargs
                                        )
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False,
                                      )
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_frame_dir (str): visual frame zip file path.
        test_mode (bool): if it's for testing.

    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_frame_dir: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_frame_dir = zip_frame_dir
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text_input tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        self.labels = None
        if not test_mode:
            self.labels = [self.anns[idx]['category_id'] for idx in range(len(self.anns))]

        # we use the standard frame_input transform as in the offifical Swin-Transformer.
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        elif num_frames <= 24:
            # step = num_frames // self.max_frame
            select_inds = list(range(0, num_frames, 1))
            select_inds = select_inds[:self.max_frame]
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            # if self.test_mode:
            #     # uniformly sample when test mode is True
            step = num_frames // self.max_frame
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frame]
            # else:
            #     # randomly sample when test mode is False
            #     select_inds = list(range(num_frames))
            #     random.shuffle(select_inds)
            #     select_inds = select_inds[:self.max_frame]
            #     select_inds = sorted(select_inds)
            # randomly sample when test mode is False
            # select_inds = list(range(0, int(num_frames / 2),2)) + list(range(int(num_frames / 2 + 1), num_frames,num_frames // 9))
            # select_inds = select_inds[:self.max_frame]

        for i, j in enumerate(select_inds):
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

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

        # text_token_type_ids = torch.zeros_like(text_input_ids)
        return text_input_ids, text_mask #, text_token_type_ids


    def tokenize_text_tfidf(self, title: str, ocr_text: str, asr_text: str) -> tuple:
        max_len = 128
        tfidf_str_title = ''

        if len(title)<2:
            tfidf_str_title = ''
        else:
            title_tfidf = count_tfidf(title + ocr_text + asr_text)
            for i in range(title_tfidf.__len__()):
                tfidf_str_title = tfidf_str_title + title_tfidf[i][0] + " "
            if len(tfidf_str_title) >= max_len/2:
                tfidf_str_title = tfidf_str_title[:(int(max_len/2))]
            if len(title) >= max_len/2:
                title = title[:(int(max_len/2))]
            tfidf_str_title = title + tfidf_str_title


        if len(ocr_text) >= max_len:
            ocr_text = ocr_text[:(int(max_len / 2))] + ocr_text[-(int(max_len / 2)):]
        if len(asr_text) >= max_len:
            asr_text = asr_text[:(int(max_len / 2))] + asr_text[-(int(max_len / 2)):]

        encoded_title = self.tokenizer(tfidf_str_title, max_length=max_len, padding='max_length', truncation=True)
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

        # text_token_type_ids = torch.zeros_like(text_input_ids)
        return text_input_ids, text_mask #, text_token_type_ids

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(idx)

        # Step 2, load title tokens
        title, asr = self.anns[idx]['title'], self.anns[idx]['asr']
        asr = re.sub('嗯{3,}', '', asr)
        ocr = sorted(self.anns[idx]['ocr'], key=lambda x: x['time'])
        ocr = ','.join([t['text'] for t in ocr])
        text_input, text_mask = self.tokenize_text_tfidf(title, ocr, asr)

        # Step 3, summarize into a dictionary
        # data = dict(
        #     frame_input=frame_input,
        #     frame_mask=frame_mask,
        #     text_input=text_input,
        #     text_mask=text_mask,
        #     # title_token_type_ids=title_token_type_ids
        # )
        data = (frame_input, frame_mask, text_input, text_mask)
        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data = data + (torch.LongTensor([label]),)

        return data


def resample(dataset):
    anns = dataset.dataset.anns
    indices = dataset.indices
    labels = [line['category_id'] for line in anns]
    label_cnt = Counter(labels)
    indices_resample = []
    for idx in indices:
        if label_cnt[anns[idx]['category_id']] < 100:
            indices_resample.extend([idx] * 5)
        elif label_cnt[anns[idx]['category_id']] < 500:
            indices_resample.extend([idx] * 3)
        elif label_cnt[anns[idx]['category_id']] < 1000:
            indices_resample.extend([idx] * 2)
        else:
            indices_resample.append(idx)

    dataset.indices = indices_resample
    print("trainset len:", len(dataset))

# 计算字符串的TF-IDF值的方法
def count_tfidf(str):
    try:
        sentences = str.split()
        sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
        document = ["".join(sent0) for sent0 in sent_words]
        tfidf_model = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b",
                                      stop_words=["是", "的", "嗯", "呀"]).fit(document)
        feature = tfidf_model.get_feature_names_out()
        # 每一行指定特征的tf-idf值
        sparse_result = tfidf_model.transform(document)
        weight = sparse_result.toarray()
    except ValueError:
        featureList = []
        return featureList

    feature_TFIDF = {}
    for i in range(len(weight)):
        for j in range(len(feature)):
            if feature[j] not in feature_TFIDF:
                feature_TFIDF[feature[j]] = weight[i][j]
            else:
                feature_TFIDF[feature[j]] = max(feature_TFIDF[feature[j]], weight[i][j])

    featureList = sorted(feature_TFIDF.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return featureList