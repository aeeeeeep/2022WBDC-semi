import json

import torch
import torch.nn as nn
import torch.nn.functional as F


from swin import swin_tiny
from category_id_map import CATEGORY_ID_LIST
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
import yaml
from functools import partial


class ALBEF(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.distill = False

        # self.visual_encoder = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        self.video_dense = nn.Linear(768, 768)
        # config = json.loads('./config.json')
        with open("./config.json", 'r') as load_f:
            config = json.load(load_f)
        config = Dict2Obj(config)
        self.embeddings = BertEmbeddings(config)
        self.text_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)

        self.cls_head = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
        )

        if self.distill:
            self.visual_backbone = swin_tiny(args.swin_pretrained_path)
            self.video_dense = nn.Linear(768, 768)
            self.text_encoder_m = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
            self.cls_head_m = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
            )

            self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                                [self.text_encoder, self.text_encoder_m],
                                [self.cls_head, self.cls_head_m],
                                ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self, frame_input, frame_mask, text_input, text_mask, text_token_type_ids, label, alpha=0.4, train=True):
        bert_embedding = self.bert(text_input, text_mask)['pooler_output']
        frame_input = self.visual_backbone(frame_input)
        frame_input = self.video_dense(frame_input)
        frame_emb = self.embeddings(inputs_embeds=frame_input)
        # frame_emb = self.visual_encoder(frame_input)
        frame_atts = torch.ones(frame_emb.size()[:-1], dtype=torch.long).to(frame_input.device)

        if train:
            output = self.text_encoder(text_token_type_ids,
                                       attention_mask=text_mask,
                                       encoder_hidden_states=frame_emb,
                                       encoder_attention_mask=frame_atts,
                                       return_dict=True
                                       )
            prediction = self.cls_head(output.last_hidden_state[:, 0, :])
            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    frame_emb_m = self.visual_encoder_m(frame_input)
                    output_m = self.text_encoder_m(text_token_type_ids,
                                                   attention_mask=text_mask,
                                                   encoder_hidden_states=frame_emb_m,
                                                   encoder_attention_mask=frame_atts,
                                                   return_dict=True
                                                   )
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:, 0, :])

                loss = (1 - alpha) * F.cross_entropy(prediction, label, label_smoothing=0.1) - alpha * torch.sum(
                    F.log_softmax(prediction, dim=1) * F.softmax(prediction_m, dim=1), dim=1).mean()
                return loss
            else:
                return self.cal_loss(prediction, label)

        else:
            output = self.text_encoder(text_token_type_ids,
                                       attention_mask=text_mask,
                                       encoder_hidden_states=frame_emb,
                                       encoder_attention_mask=frame_atts,
                                       return_dict=True
                                       )
            prediction = self.cls_head(output.last_hidden_state[:, 0, :])
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

class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x

class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding

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