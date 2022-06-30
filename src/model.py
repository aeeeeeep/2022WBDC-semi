import torch
import torch.nn as nn
import torch.nn.functional as F
from category_id_map import CATEGORY_ID_LIST

# from masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        # encoder后特征
        self.bert = Bert_encoder.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        # mean pooling
        self.mean_pooling = MeanPooling()

        self.drop = nn.Dropout(p=0.2)
        # 线性层
        self.classify_dense = nn.Linear(768, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):
        encoder_outputs, mask = self.bert(inputs)
        # output = self.mean_pooling(encoder_outputs, mask)
        # 另一种mean pooling实现方案
        output = torch.einsum("bsh,bs,b->bh", encoder_outputs, mask.float(), 1 / mask.float().sum(dim=1) + 1e-9)
        output = self.drop(output)
        prediction = self.classify_dense(output)

        if inference:
            # return torch.argmax(prediction, dim=1)
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label, label_smoothing=0.1)  # label smooth

        # loss_func = FocalLoss(reduction="mean", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # loss = loss_func(prediction.view(-1, 200), label.view(-1))  # 损失函数更换为FocalLoss

        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class Bert_encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

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

    def forward(self, inputs):
        frame_input, frame_mask, frame_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs[
            'frame_token_type_ids']
        text_input, text_mask, text_token_type_ids = inputs['text_input'], inputs['text_mask'], inputs[
            'text_token_type_ids']

        text_emb = self.embeddings(input_ids=text_input)
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]

        frame_input = self.video_dense(frame_input)
        # frame_input = self.video_activation(frame_input)
        frame_emb = self.embeddings(inputs_embeds=frame_input)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, frame_emb, text_emb], 1)

        mask = torch.cat([cls_mask, frame_mask, text_mask], 1)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_mask)['last_hidden_state']
        return encoder_outputs, mask


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
