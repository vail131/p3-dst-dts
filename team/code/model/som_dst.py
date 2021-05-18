"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_bert import BertEmbeddings

class CustomBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, slot_token_id):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.slot_token_id = slot_token_id

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size = input_ids.size(0)
        position_embeddings = self.position_embeddings(position_ids).expand(batch_size, -1, -1)
        first_pos = torch.where(input_ids == self.slot_token_id)[1][::45]
        tmp = torch.zeros_like(position_embeddings)
        tmp[torch.arange(tmp.size(0)), first_pos] = 1
        tmp = tmp.cumsum(dim=1).to(torch.bool)
        position_embeddings = position_embeddings.masked_fill(tmp, 0)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SomDST(BertPreTrainedModel):
    def __init__(self, config, n_op, n_domain, update_id, len_tokenizer, exclude_domain=False,
            slot_token_id=35000):
        super(SomDST, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.encoder = Encoder(config, n_op, n_domain, update_id, exclude_domain, slot_token_id)
        self.encoder.bert.resize_token_embeddings(len_tokenizer)
        self.decoder = Decoder(config, self.encoder.bert.embeddings.word_embeddings.weight)
        # self.apply(self.init_parameter)
        self.init_weights()

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                max_value, op_ids=None, max_update=None, teacher=None):
        # B: batch size
        # S: seq len
        # H: hidden size
        # J: total slot
        # MU: max update
        # MV: max value
        # NOP: num op
        # ND: num domain

        # input_ids: B S
        # token_type_ids: B S
        # state_positions: B J
        # attention_mask: B S
        # op_ids: B J
        # teacher: B MU MV
        
        enc_outputs = self.encoder(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   state_positions=state_positions,
                                   attention_mask=attention_mask,
                                   op_ids=op_ids,
                                   max_update=max_update)

        # domain_scores: B MU H
        # state_scores: B J NOP
        # decoder_inputs: B MU H
        # sequence_output: B S H
        # pooled_output: 1 B H
        domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs

        gen_scores = self.decoder(input_ids, decoder_inputs, sequence_output,
                                  pooled_output, max_value, teacher) # B MU MV VB

        # domain_scores: B MU H
        # state_scores: B J NOP
        # gen_scores: B MU MV VB
        return domain_scores, state_scores, gen_scores

    # @staticmethod
    # def init_parameter(module):
    #     # if isinstance(module, nn.Linear):
    #     #     torch.nn.init.xavier_normal_(module.weight)
    #     #     torch.nn.init.constant_(module.bias, 0.0)
    #     if isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
    #         torch.nn.init.xavier_normal_(module.weight_ih_l0)
    #         torch.nn.init.xavier_normal_(module.weight_hh_l0)
    #         torch.nn.init.constant_(module.bias_ih_l0, 0.0)
    #         torch.nn.init.constant_(module.bias_hh_l0, 0.0)


class Encoder(nn.Module):
    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False, slot_token_id=35000):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.exclude_domain = exclude_domain
        self.bert = BertModel(config)
        # bert_config = BertConfig.from_pretrained(config.model_name_or_path)
        # custom_emb = CustomBertEmbeddings(bert_config, slot_token_id)
        # custom_emb.word_embeddings = self.bert.embeddings.word_embeddings
        # custom_emb.position_embeddings = self.bert.embeddings.position_embeddings
        # custom_emb.token_type_embeddings = self.bert.embeddings.token_type_embeddings
        # self.bert.embeddings = custom_emb

        self.dropout = nn.Dropout(0.1)
        self.action_cls = nn.Linear(config.hidden_size, n_op)
        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(config.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                op_ids=None, max_update=None):
        # B: batch size
        # S: seq len
        # H: hidden size
        # J: total slot
        # MU: max update
        # MV: max value
        # NOP: num op
        # ND: num domain

        # input_ids: B S
        # token_type_ids: B S
        # state_positions: B J
        # attention_mask: B S
        # op_ids: B J

        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)

        # sequence_output: B S H
        # pooled_output: B H
        sequence_output, pooled_output = bert_outputs[:2]

        # B J H
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        # B J H
        state_output = torch.gather(sequence_output, 1, state_pos)

        # B J NOP
        state_scores = self.action_cls(self.dropout(state_output))  # B,J,4
        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # 1
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output)) # B ND

        batch_size = state_scores.size(0)
        if op_ids is None: 
            # B J
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()

        gathered = []
        # b: J H
        # a: J
        # UP: current update cnt
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
            if a.sum().item() != 0:
                # 1 UP H
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1*gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1) # 1 MU H
            else:
                # 1 MU H
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            gathered.append(v)
        # B MU H
        decoder_inputs = torch.cat(gathered)

        # domain_scores: B MU H
        # state_scores: B J NOP
        # decoder_inputs: B MU H
        # sequence_output: B S H
        # pooled_output: 1 B H
        return domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = bert_model_embedding_weights.size(0)
        self.embed = nn.Embedding(35004, config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size*3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

        for n, p in self.gru.named_parameters():
            if 'weight' in n:
                p.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len, teacher=None):
        # B: batch size
        # S: seq len
        # H: hidden size
        # J: total slot
        # MU: max update
        # MV: max value
        # NOP: num op
        # ND: num domain
        # VB: vocab size

        # x: B S
        # decoder_inputs: B MU H
        # encoder_output: B S H
        # hidden: 1 B H
        # teacher: B MU MV

        # B S
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        # MU B MV VB
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update):
            w = state_in[:, j].unsqueeze(1)  # B 1 H
            slot_value = []
            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B, H
                # B,S,H * B,H,1 
                attn_history = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,S, 1
                attn_history = attn_history.squeeze(-1).masked_fill(mask, -1e9) # B S
                attn_history = nn.functional.softmax(attn_history, -1)  # B S

                # B,H * H,VB => B,VB
                attn_vocab = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B VB
                attn_vocab = nn.functional.softmax(attn_vocab, -1)

                # B,1,S * B, S,H => B,1,H
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,H
                # B 1 3*H   *  3*H 1 -> B 1 1
                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))
                p_gen = p_gen.squeeze(-1) # B 1

                # B, VB
                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,VB
                _, w_idx = p_final.max(-1) # B
                slot_value.append([ww.tolist() for ww in w_idx]) # B list but each list size 1
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1) # B 1  D 
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                all_point_outputs[j, :, k, :] = p_final

        # B MU MV VB
        return all_point_outputs.transpose(0, 1)

        