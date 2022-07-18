import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertModel

from utils.swin import swin_tiny
from utils.modeling import LXRTEncoder, LXRTModel, LXRTFeatureExtraction
from category_id_map import CATEGORY_ID_LIST


class LXMERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        # self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
        #                          output_size=args.vlad_hidden_size, dropout=args.dropout)
        # self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        self.video_dense = nn.Linear(768, 768)

        self.encoder = LXRTFeatureExtraction.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)

        # self.fusion = ConcatDenseSE(768, 768, args.se_ratio, args.dropout)
        # self.encoder = Bert_encoder.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        # self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))
        # self.classifier = nn.Sequential(
        #           nn.Linear(768, 768),
        #           nn.ReLU(),
        #           nn.Linear(768, len(CATEGORY_ID_LIST))
        #         )
        self.drop = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))

    def forward(self, frame_input, frame_mask, text_input, text_mask, label=1, inference=False):
        frame_inputs = self.visual_backbone(frame_input)

        # frame_fea = self.nextvlad(frame_inputs, frame_mask)
        # frame_fea = self.enhance(frame_fea)
        frame_fea = self.video_dense(frame_inputs)

        encoder_outputs = self.encoder(input_ids=text_input, attention_mask=text_mask, visual_feats=frame_fea, visual_attention_mask=frame_mask)
        output = self.drop(encoder_outputs)
        prediction = self.classifier(output)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, label)

    # @torch.no_grad()
    # def copy_params(self):
    #     for model_pair in self.model_pairs:
    #         for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
    #             param_m.data.copy_(param.data)  # initialize
    #             param_m.requires_grad = False  # not update by gradient
    #
    # @torch.no_grad()
    # def _momentum_update(self):
    #     for model_pair in self.model_pairs:
    #         for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
    #             param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label, label_smoothing=0.1)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad

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

class Bert_encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.lxmert_encoder = LXRTEncoder(config)
        self.text_encoder = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)


        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, frame_fea, frame_mask, text_input, text_mask):
        text_emb = self.embeddings(input_ids=text_input)
        frame_emb = self.embeddings(inputs_embeds=frame_fea)
        text_feats, frame_feats = self.lxmert_encoder(text_emb, text_mask, frame_emb, frame_mask)

        return encoder_outputs

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
            return

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