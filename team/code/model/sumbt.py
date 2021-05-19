"""
Most of code is from https://github.com/SKTBrain/SUMBT
"""
import math
import os.path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForPreTraining, AutoConfig

# https://github.com/CyberZHG/torch-position-embedding
from torch_position_embedding import PositionEmbedding 

class AutoForUtteranceEncoding(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.base = model

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = AutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        ret = cls(config, model)
        return ret

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.to(dtype=scores.dtype)
            mask = (1.0 - mask) * -10000.0
            scores = scores + mask
            
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores

class Transformer(nn.Module):
    def __init__(self, attn_head, output_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(attn_head, output_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x, mask, use_no_lookup=False):
        bs = x.size(0)
        seq = x.size(1)
        if use_no_lookup:
            if mask is None:
                mask = torch.ones(bs, 1, seq, dtype=torch.long).to(x.device)
            mask = torch.tril(mask.repeat(1, seq, 1))

        tmp = self.layer_norm1(x)
        sub_layer = self.attn(tmp, tmp, tmp, mask)
        x = x + self.dropout(sub_layer)

        tmp = self.layer_norm2(x)
        sub_layer = self.ffwd(tmp)
        x = x + self.dropout(sub_layer)

        return x

# class Transformer(nn.Module):
#     def __init__(self, attn_head, output_dim, dropout=0.1):
#         super().__init__()
#         self.attn = MultiHeadAttention(attn_head, output_dim, dropout)
#         self.layer_norm1 = nn.LayerNorm(output_dim)
#         self.layer_norm2 = nn.LayerNorm(output_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.ffwd = nn.Sequential(
#             nn.Linear(output_dim, output_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(output_dim, output_dim),
#         )

#     def forward(self, x, mask):
#         tmp = self.layer_norm1(x)
#         sub_layer = self.attn(tmp, tmp, tmp, mask)
#         x = x + self.dropout(sub_layer)

#         tmp = self.layer_norm2(x)
#         sub_layer = self.ffwd(tmp)
#         x = x + self.dropout(sub_layer)

#         return x

class SUMBT(nn.Module):
    def __init__(self, args, num_labels, device, use_no_lookup=False):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.rnn_num_layers = args.num_rnn_layers
        self.zero_init_rnn = args.zero_init_rnn
        self.max_seq_length = args.max_seq_length
        self.max_label_length = args.max_label_length
        self.num_labels = num_labels
        self.num_slots = len(num_labels)
        self.attn_head = args.attn_head
        self.use_larger_slot_encoding = args.use_larger_slot_encoding
        self.use_transformer = args.use_transformer
        self.device = device
        self.use_no_lookup = use_no_lookup

        ### Utterance Encoder
        self.utterance_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        try:
            self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        except AttributeError:
            self.hidden_dropout_prob = 0.1

        if args.fix_utterance_encoder:
            for p in self.utterance_encoder.base.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = AutoForUtteranceEncoding.from_pretrained(
            args.model_name_or_path
        )
        # os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        for p in self.sv_encoder.base.parameters():
            p.requires_grad = False

        if self.use_larger_slot_encoding:
            self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim*self.max_label_length)
            self.slot_pooler = nn.Linear(self.max_label_length * self.bert_output_dim, self.bert_output_dim)
        else:
            self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)

        self.value_lookup = nn.ModuleList(
            [nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels]
        )

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)
        
        if self.use_transformer:
            self.transformers = nn.ModuleList(
                [Transformer(self.attn_head, self.bert_output_dim, dropout=0) for _ in range(args.n_transformers)]
            )

            self.position_emb = PositionEmbedding(num_embeddings=50,
                embedding_dim=self.bert_output_dim,
                mode=PositionEmbedding.MODE_ADD)
        else:
            if self.rnn_num_layers == 1:
                gru_dropout_prob = 0
            else:
                gru_dropout_prob = self.hidden_dropout_prob
            self.nbt = nn.GRU(
            input_size=self.bert_output_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_num_layers,
            dropout=gru_dropout_prob,
            batch_first=True,
            )
            self.init_parameter(self.nbt)

            if not self.zero_init_rnn:
                self.rnn_init_linear = nn.Sequential(
                    nn.Linear(self.bert_output_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.hidden_dropout_prob),
                )

            self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
            self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        # self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):
        self.sv_encoder.eval()

        # Slot encoding
        pbar = tqdm(desc='Making slot encoding -- waiting...', bar_format='{desc} -> {elapsed}')
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(
            slot_ids.device
        )
        slot_mask = slot_ids > 0
        hid_slot = self.sv_encoder(
            slot_ids.view(-1, self.max_label_length),
            slot_type_ids.view(-1, self.max_label_length),
            slot_mask.view(-1, self.max_label_length),
        ).hidden_states[-1]
        
        if self.use_larger_slot_encoding:
            hid_slot = hid_slot.masked_fill(slot_mask.unsqueeze(-1) == False, 0)
            hid_slot = hid_slot.view(self.num_slots, -1)
        else:
            hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
    
        old = self.slot_lookup
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)
        assert self.slot_lookup.weight.shape == old.weight.shape, f'{self.slot_lookup.weight.shape} {old.weight.shape}'
        pbar.set_description('Making slot encoding -- DONE')
        pbar.close()

        pbar = tqdm(enumerate(label_ids), desc='Making value lookup', total=len(label_ids))
        for s, label_id in pbar:
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(
                label_id.device
            )
            label_mask = label_id > 0
            hid_label = self.sv_encoder(
                label_id.view(-1, self.max_label_length),
                label_type_ids.view(-1, self.max_label_length),
                label_mask.view(-1, self.max_label_length),
            ).hidden_states[-1]
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()

            old = self.value_lookup[s]
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1
            assert self.value_lookup[s].weight.shape == old.weight.shape, f'{self.value_lookup[s].weight.shape} {old.weight.shape}'
        pbar.close()
        self.sv_encoder = None

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        target_slot=None,
    ):
        # B = Batch Size
        # M = Max Turn Size
        # N = Seq Len
        # J = Target_slot Len
        # H_GRU = RNN Hidden Dim
        # L = Label Len
        
        # input_ids: [B, M, N]
        # token_type_ids: [B, M, N]
        # attention_mask: [B, M, N]
        # labels: [B, M, J]

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            target_slot = list(range(0, self.num_slots))

        ds = input_ids.size(0)  # Batch size (B)
        ts = input_ids.size(1)  # Max turn size (M)
        bs = ds * ts # B * M
        slot_dim = len(target_slot)  # J

        # Utterance encoding
        hidden = self.utterance_encoder(
            input_ids.view(-1, self.max_seq_length),
            token_type_ids.view(-1, self.max_seq_length),
            attention_mask.view(-1, self.max_seq_length),
        ).hidden_states[-1] # [B*M, N, H]

        hidden = torch.mul(
            hidden,
            attention_mask.view(-1, self.max_seq_length, 1)
            .expand(hidden.size())
            .float(),
        )
        hidden = hidden.repeat(slot_dim, 1, 1)  # [J*M*B, N, H]

        hid_slot = self.slot_lookup.weight[
            target_slot, :
        ]  # Select target slot embedding
        if self.use_larger_slot_encoding: # [J, H * L]
            hid_slot = hid_slot.view(slot_dim, self.max_label_length, self.bert_output_dim) # [J, L, H]
            hid_slot = hid_slot.repeat(1, bs, 1).view(bs * slot_dim, self.max_label_length, -1) # [J*M*B, L, H]
        else: # [J, H]
            hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [J*M*B, H]

        # Attended utterance vector
        hidden = self.attn(
            hid_slot,  # q^s  [J*M*B, L, H]
            hidden,  # U [J*M*B, N, H]
            hidden,  # U [J*M*B, N, H]
            mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1),
        ) # [J*M*B, L, H]

        if self.use_larger_slot_encoding: # [J*M*B, H]
            hidden = self.slot_pooler(hidden.view(-1, self.max_label_length * self.bert_output_dim))
        else:
            hidden = hidden.squeeze()  # h [J*M*B, H] Aggregated Slot Context

        hidden = hidden.view(slot_dim, ds, ts, -1).view(
            -1, ts, self.bert_output_dim
        )  # [J*B, M, H]

        if self.use_transformer:
            hidden = self.position_emb(hidden)

            transformer_mask = torch.any(attention_mask, -1).repeat(slot_dim, 1).unsqueeze(1)
            for transformer in self.transformers:
                hidden = transformer(
                    hidden,
                    mask=transformer_mask, # [J*B, 1, M]
                    use_no_lookup=self.use_no_lookup,
                ) # [J*B, M, H]
        else:
            # NBT       
            if self.zero_init_rnn:
                h = torch.zeros(
                    self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
                ).to(
                    self.device
                )  # [1, slot_dim*ds, hidden]
            else:
                h = hidden[:, 0, :].unsqueeze(0).expand(self.rnn_num_layers, -1, -1) # 원래 repeat(self.rnn_num_layers, 1, 1) 이었는데 바꿈
                h = self.rnn_init_linear(h)    # 왜냐면 1 dim은 언제나 사이즈 1이니까 expand 사용하는게 더 좋음(expand 메모리 복사 없음, repeat 복사됨)

            if isinstance(self.nbt, nn.GRU):
                rnn_out, _ = self.nbt(hidden, h)  # [J*B, M, H_GRU]
            elif isinstance(self.nbt, nn.LSTM):
                c = torch.zeros(
                    self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim
                ).to(
                    self.device
                )  # [1, slot_dim*ds, hidden]
                rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
            hidden = self.layer_norm(self.linear(self.dropout(rnn_out)))

        hidden = hidden.view(slot_dim, ds, ts, -1) # [J, B, M, H]

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = (
                hid_label.unsqueeze(0)
                .unsqueeze(0)
               .repeat(ds, ts, 1, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _hidden = (
                hidden[s, :, :, :]
                .unsqueeze(2)
                .repeat(1, 1, num_slot_labels, 1)
                .view(ds * ts * num_slot_labels, -1)
            )
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)
            _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            # if labels is not None:
            #     _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
            #     loss_slot.append(_loss.item())
            #     loss += _loss

        pred_slot = torch.cat(pred_slot, 2)
        
        return output, pred_slot

        # # calculate joint accuracy
        # accuracy = (pred_slot == labels).view(-1, slot_dim)
        # acc_slot = (
        #     torch.sum(accuracy, 0).float()
        #     / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        # )
        # acc = (
        #     sum(torch.sum(accuracy, 1) / slot_dim).float()
        #     / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()
        # )  # joint accuracy

        # return loss, loss_slot, acc, acc_slot, pred_slot
        

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)