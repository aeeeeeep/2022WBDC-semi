import json

import torch
import torch.nn as nn
import torch.nn.functional as F


from swin import swin_tiny
# from cswin import CSWin_64_12211_tiny_224
from category_id_map import CATEGORY_ID_LIST
# from transformers import BertModel, BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
import yaml
from functools import partial


class ALBEF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.distill = False
        self.bert = Bert_encoder.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        # self.mean_pooling = MeanPooling()
        self.drop = nn.Dropout(p=0.2)
        self.cls_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, len(CATEGORY_ID_LIST))
        )

        if self.distill:
            self.bert_m = Bert_encoder.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
            # self.mean_pooling_m = MeanPooling()
            self.drop_m = nn.Dropout(p=0.2)
            self.cls_head_m = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, len(CATEGORY_ID_LIST))
            )

            self.model_pairs = [[self.bert, self.bert_m],
                                [self.cls_head, self.cls_head_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self, frame_input, frame_mask, text_input, text_mask, label, alpha=0.4, train=True):
        if train:
            encoder_outputs, mask = self.bert(frame_input, frame_mask, text_input, text_mask)
            output = torch.einsum("bsh,bs,b->bh", encoder_outputs, mask.float(), 1 / mask.float().sum(dim=1) + 1e-9)
            output = self.drop(output)
            prediction = self.cls_head(output)
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    encoder_outputs_m, mask_m = self.bert_m(frame_input, frame_mask, text_input, text_mask)
                    output_m = torch.einsum("bsh,bs,b->bh", encoder_outputs_m, mask_m.float(), 1 / mask_m.float().sum(dim=1) + 1e-9)
                    output_m = self.drop_m(output_m)
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:, 0, :])

                label = label.squeeze(dim=1)
                loss = (1 - alpha) * F.cross_entropy(prediction, label, label_smoothing=0.1) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1), dim=1).mean()
                return loss
            else:
                return self.cal_loss(prediction, label)

        else:
            encoder_outputs, mask = self.bert(frame_input, frame_mask, text_input, text_mask)
            output = torch.einsum("bsh,bs,b->bh", encoder_outputs, mask.float(), 1 / mask.float().sum(dim=1) + 1e-9)
            output = self.drop(output)
            prediction = self.cls_head(output)
            return torch.argmax(prediction, dim=1)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label, label_smoothing=0.1)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class Dict2Obj(dict):

    def __getattr__(self, key):
        print('getattr is called')
        if key not in self:
            return None
        else:
            value = self[key]
            if isinstance(value,dict):
                value = Dict2Obj(value)
            return value

class Bert_encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.visual_backbone = swin_tiny('opensource_models/swin_tiny_patch4_window7_224.pth')
        self.video_dense = nn.Linear(768, 768)
        # self.video_activation = nn.Tanh()
        # self.video_embeddings = BertEmbeddings(config)
        # self.video_embeddings = self.embeddings

        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, frame_input, frame_mask, text_input, text_mask):

        text_emb = self.embeddings(input_ids=text_input)
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]

        frame_input_ = self.visual_backbone(frame_input)
        frame_input_ = self.video_dense(frame_input_)
        frame_emb = self.embeddings(inputs_embeds=frame_input_)
        # frame_input = self.video_activation(frame_input)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, frame_emb, text_emb], 1)

        mask = torch.cat([cls_mask, frame_mask, text_mask], 1)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_mask)['last_hidden_state']
        return encoder_outputs, mask