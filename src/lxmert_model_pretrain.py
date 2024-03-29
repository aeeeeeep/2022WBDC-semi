import sys
sys.path.append("..")
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertModel, BertConfig
from transformers import BertTokenizer

from utils.swin import swin_tiny
# from utils.deit import deit_base_patch16_LS as deit
from utils.masklm import MaskLM, MaskVideo, ShuffleVideo
from utils.modeling import LXRTEncoder, LXRTModel, LXRTFeatureExtraction, LXRTPretraining
from category_id_map import CATEGORY_ID_LIST


class LXMERT_PRE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            args.bert_dir,
            cache_dir=args.bert_cache,
            do_lower_case=True
        )
        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        self.video_dense = nn.Linear(768, 768)

        # mlm
        self.lm = MaskLM(tokenizer_path=args.bert_dir)
        self.num_class = 10000
        self.vocab_size = 21128

        # itm
        self.sv = ShuffleVideo()
        self.newfc_itm_cls = torch.nn.Linear(768, 1)

        self.encoder = LXRTPretraining.from_pretrained(args.bert_dir,
                                                       cache_dir=args.bert_cache,
                                                       task_mask_lm=True,
                                                       )

    def forward(self, frame_input, frame_mask, text_input, text_mask):
        with torch.no_grad():
            frame_inputs = self.visual_backbone(frame_input)
        frame_fea = self.video_dense(frame_inputs)

        # mlm
        input_ids, lm_label = self.lm.torch_mask_tokens(text_input.cpu())
        text_input_ids = input_ids.to(text_input.device)
        lm_label = lm_label.to(text_input_ids.device)

        # itm
        input_feature, video_text_match_label = self.sv.torch_shuf_video(frame_fea.cpu())
        frame_feature = input_feature.to(frame_fea.device)
        video_text_match_label = video_text_match_label.to(frame_feature.device)

        lm_prediction_scores, pooled_output = self.encoder(input_ids=text_input_ids,
                                                           token_type_ids=None,
                                                           attention_mask=text_mask,
                                                           frame_mask=frame_mask,
                                                           visual_feats=frame_feature
                                                           )

        # mlm
        mlm_pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
        mlm_loss = nn.CrossEntropyLoss()(mlm_pred, lm_label.view(-1))
        mlm_accuracy = torch.sum(lm_prediction_scores.argmax(dim=-1).view(-1) == lm_label.view(-1)) / (
                torch.sum(lm_label.view(-1) > 0) + 1e-12)
        # mlm_loss = torch.log(mlm_loss + 1e-12)

        # itm
        itm_pred = self.newfc_itm_cls(pooled_output)
        loss_itm = nn.BCEWithLogitsLoss()(itm_pred.view(-1), video_text_match_label.view(-1))
        itm_accuracy = torch.sum((itm_pred.view(-1) > 0.5).int() == video_text_match_label.view(-1).int()).float() / \
                       itm_pred.view(-1).shape[0]
        # loss_itm += torch.log(loss_itm + 1e-12)

        return mlm_loss, loss_itm, mlm_accuracy, itm_accuracy



