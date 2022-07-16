import torch
import torch.nn as nn
import torch.nn.functional as F

from swin import swin_tiny
from functools import partial
from transformers import BertTokenizer, BertConfig
from xbert import BertForMaskedLM


class ALBEF_PRE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)

        bert_config = BertConfig.from_json_file('./config.json')
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_config)
        self.text_encoder = BertForMaskedLM.from_pretrained(args.bert_dir, cache_dir=args.bert_cache, config=bert_config)

        self.vision_proj = nn.Linear(args.vlad_hidden_size, 768)
        self.text_proj = nn.Linear(768, 768)

        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.queue_size = 65536
        self.momentum = 0.995
        self.itm_head = nn.Linear(768, 2)

        # create momentum models
        self.nextvlad_m = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                   output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance_m = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(args.bert_dir, cache_dir=args.bert_cache,
                                                              config=bert_config)
        self.vision_proj_m = nn.Linear(args.vlad_hidden_size, 768)
        self.text_proj_m = nn.Linear(768, 768)

        self.model_pairs = [[self.nextvlad, self.nextvlad_m],
                            [self.enhance, self.enhance_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("frame_queue", torch.randn(96, self.queue_size))
        self.register_buffer("text_queue", torch.randn(96, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.frame_queue = nn.functional.normalize(self.frame_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


    def forward(self, frame_input, frame_mask, text_input, text_mask, alpha=0.4):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        frame_backbone = self.visual_backbone(frame_input)
        frame_emb = self.nextvlad(frame_backbone, frame_mask)
        frame_emb = self.enhance(frame_emb)

        frame_feat = F.normalize(self.vision_proj(frame_emb),dim=-1)

        text_output = self.text_encoder.bert(text_input, attention_mask=text_mask,
                                             # return_dict=True, mode='text')
                                             return_dict=True)
        text_emb = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_emb),dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            frame_emb_m = self.nextvlad_m(frame_backbone, frame_mask)
            frame_emb_m = self.enhance_m(frame_emb_m)
            frame_feat_m = F.normalize(self.vision_proj_m(frame_emb_m), dim=-1)
            frame_feat_all = torch.cat([frame_feat_m.t(), self.frame_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m.bert(text_input, attention_mask=text_mask,
                                                    # return_dict=True, mode='text')
                                                     return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = frame_feat_all @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ frame_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(frame_input.device)
            sim_targets.fill_diagonal_(1)

            # sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            # sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = frame_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ frame_feat_all / self.temp
        #
        # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        #
        # loss_ita = (loss_i2t + loss_t2i) / 2
        #
        # self._dequeue_and_enqueue(frame_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        output_pos = self.text_encoder.bert(encoder_embeds=text_emb,
                                            attention_mask=text_mask,
                                            encoder_hidden_states=frame_emb,
                                            encoder_attention_mask=frame_mask,
                                            return_dict=True,
                                            mode='fusion',
                                            )
        with torch.no_grad():
            bs = frame_input.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        frame_emb_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            frame_emb_neg.append(frame_emb[neg_idx])
        frame_emb_neg = torch.stack(frame_emb_neg, dim=0)

        # select a negative text for each image
        text_emb_neg = []
        text_mask_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_emb_neg.append(text_emb[neg_idx])
            text_mask_neg.append(text_mask[neg_idx])
        text_emb_neg = torch.stack(text_emb_neg, dim=0)
        text_mask_neg = torch.stack(text_mask_neg, dim=0)

        text_emb_all = torch.cat([text_emb, text_emb_neg], dim=0)
        text_mask_all = torch.cat([text_mask, text_mask_neg], dim=0)

        frame_emb_all = torch.cat([frame_emb_neg, frame_emb], dim=0)
        frame_mask_all = torch.cat([frame_mask, frame_mask], dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds=text_emb_all,
                                            attention_mask=text_mask_all,
                                            encoder_hidden_states=frame_emb_all,
                                            encoder_attention_mask=frame_mask_all,
                                            return_dict=True,
                                            mode='fusion',
                                            )

        vl_emb = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_emb)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(frame_input.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= MLM ========================##
        input_ids = text_input.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, frame_input.device, targets=labels,
                                      probability_matrix=probability_matrix)

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text_mask,
                                           encoder_hidden_states=frame_emb_m,
                                           encoder_attention_mask=frame_mask,
                                           return_dict=True,
                                           return_logits=True,
                                           )
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text_mask,
                                       encoder_hidden_states=frame_emb,
                                       encoder_attention_mask=frame_mask,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha
                                       )
        loss_mlm = mlm_output.loss

        # return loss_mlm, loss_ita, loss_itm
        return loss_mlm, loss_itm

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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.frame_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

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
