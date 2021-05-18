from data_utils import DSTPreprocessor, _truncate_seq_pair, convert_state_dict
from tqdm.auto import tqdm
import torch
import random
import numpy as np
from copy import deepcopy

from dataclasses import dataclass
from typing import List, Optional, Union
from functools import partial

flatten = lambda x: [i for s in x for i in s]

OP_SET = {
    '2': {'update': 0, 'carryover': 1},
    '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3},
    '6': {'delete': 0, 'update': 1, 'dontcare': 2, 'carryover': 3, 'yes': 4, 'no': 5}
}

@dataclass
class SOMDSTFeature:
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    state_position_ids: List[int]

    # 불확실
    op_ids: List[int]
    domain_id: int
    gen_ids: List[List[int]]


def get_op_id(before_value, value, opcode):
    cur_op_set = OP_SET[opcode]

    if before_value == value:
        return cur_op_set['carryover']
    
    if value == 'none' and 'delete' in cur_op_set:
        return cur_op_set['delete']
    if value == 'yes' and 'yes' in cur_op_set:
        return cur_op_set['yes']
    if value == 'no' and 'no' in cur_op_set:
        return cur_op_set['no']
    if value == 'dontcare' and 'dontcare' in cur_op_set:
        return cur_op_set['dontcare']
    
    return cur_op_set['update']


# Batch Stuff: max_value, max_update

# TODO: special token add to tokenizer, [SLOT], [A-U], [S-V] [NULL]
# RESIZE MODEL EMB
# Assume example is SYS - USR order
# SLOT DATA MUST be SHUFFLED -> currently not
class SOMDSTPreprocessor(DSTPreprocessor):
    extra_special_tokens = ['[SLOT]', '[A-U]', '[S-V]', '[NULL]']
    eos_token = '[EOS]'
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None,
        max_seq=512, opcode='4'):
        src_tokenizer.add_special_tokens(
            {'additional_special_tokens': self.extra_special_tokens}
            )
        src_tokenizer.eos_token = self.eos_token

        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.max_seq = max_seq
        self.opcode = opcode
        self.op_code_type = OP_SET[self.opcode]
        self.idx2code = {v:k for k, v in self.op_code_type.items()}

        self.domain_meta = sorted(list(set(x.split('-')[0] for x in self.slot_meta)))
        self.slot_domain_info = {x:self.domain_meta.index(x.split('-')[0]) for x in self.slot_meta}

        self.slot_token_id = self.src_tokenizer.encode('[SLOT]', add_special_tokens=False)[0]
        self.s_v_token_id = self.src_tokenizer.encode('[S-V]', add_special_tokens=False)[0]
        self.null_token_id = self.src_tokenizer.encode('[NULL]', add_special_tokens=False)[0]

        self.slot_name_tokenized = {k: self.src_tokenizer.encode(k, add_special_tokens=False) 
                        for k in self.slot_meta}

    def _convert_example_to_feature(self, example, which=''):
        guid = example.guid

        label = example.label
        if label is None:
            label = {}
        states = convert_state_dict(label)
        before_states = convert_state_dict(example.before_label)
        tok_slots = []
        val_tok_slots = []
        op_ids = []
        domain_score = [0] * len(self.domain_meta)
        gen_ids = []
        for i in range(len(domain_score)):
            domain_score[i] += random.random() / 4
        for slot in self.slot_meta:
            value = states.get(slot, "none")
            before_value = before_states.get(slot, 'none')
            op_id = get_op_id(before_value, value, self.opcode)
            op_ids.append(op_id)
            domain_score[self.slot_domain_info[slot]] += OP_SET[self.opcode]['carryover'] != op_id
            if value == 'none':
                value = '[NULL]'
        
            tok_slot_data = self.slot_name_tokenized[slot]
            tok_value = self.src_tokenizer.encode(value, add_special_tokens=False)
            tok_slot = [self.slot_token_id] + tok_slot_data + [self.s_v_token_id] + tok_value
            val_tok_slot = [self.slot_token_id] + tok_slot_data + [self.s_v_token_id, self.null_token_id]

            tok_slots.extend(tok_slot)
            val_tok_slots.extend(val_tok_slot)

            if OP_SET[self.opcode]['update'] == op_id:
                gen_ids.append(tok_value + [self.src_tokenizer.eos_token_id])

        domain_id = np.argmax(domain_score)
        history = example.context_turns[-2:]
        if len(history) == 0:
            history = ['', '']
        concat_history = history[0] + '[A-U]' + history[1]
        tok_history = self.src_tokenizer.encode(concat_history, add_special_tokens=False)

        current = example.current_turn[0] + '[A-U]' + example.current_turn[1]
        tok_current = self.src_tokenizer.encode(current, add_special_tokens=False)

        max_seq_len = self.max_seq - 3

        if which == 'val':
            tok_slots = val_tok_slots
        slots_len = min(len(tok_slots), max_seq_len)
        current_len = min(max_seq_len - slots_len, len(tok_current))
        history_len = min(max_seq_len - slots_len - current_len, len(tok_history))
        
        history_start = len(tok_history) - history_len
        tok_history = [self.src_tokenizer.cls_token_id] + tok_history[history_start:] + \
                [self.src_tokenizer.sep_token_id]

        current_start = len(tok_current) - current_len
        tok_current = tok_current[current_start:] + [self.src_tokenizer.sep_token_id]

        assert len(tok_slots) == slots_len

        input_ids = tok_history + tok_current + tok_slots
        segment_ids = [0] * len(tok_history) + [1] * len(tok_current) + [1] * len(tok_slots)
        
        state_position_ids = []
        for i, v in enumerate(input_ids):
            if v == self.slot_token_id:
                state_position_ids.append(i)
        
        return SOMDSTFeature(
            guid=guid,
            input_ids=input_ids,
            segment_ids=segment_ids,
            state_position_ids=state_position_ids,
            op_ids=op_ids,
            domain_id=domain_id,
            gen_ids=gen_ids,
        )

    def convert_examples_to_features(self, examples, which=''):
        tdata = tqdm(examples, desc=f'Converting {which} examples to features')
        return list(map(partial(self._convert_example_to_feature, which=which), tdata))


    # state_scores: J NOP
    # gen_scores: MU MV VB
    def recover_state(self, before_states, state_score, gen_score):
        _, state_actions = state_score.max(-1)

        update_idx = 0
        new_states = {}
        for i, k in enumerate(self.slot_meta):
            sa_index = state_actions[i].item()
            if self.idx2code[sa_index] == 'update':
                if gen_score.size(1) == 0:
                    value = 'none'
                else:
                    value = self.src_tokenizer.decode(gen_score[update_idx].max(-1)[1].cpu(), skip_special_tokens=True)
                new_states[k] = value
                update_idx += 1
            elif self.idx2code[sa_index] == 'delete':
                pass
            elif self.idx2code[sa_index] == 'dontcare':
                new_states[k] = 'dontcare'
            elif self.idx2code[sa_index] == 'carryover':
                if k in before_states:
                    new_states[k] = before_states[k]
            elif self.idx2code[sa_index] == 'yes':
                new_states[k] = 'yes'
            elif self.idx2code[sa_index] == 'no':
                new_states[k] = 'no'
            else:
                raise NotImplementedError()
            
        return new_states

    def collate_fn(self, batch):
        input_ids = torch.LongTensor(self.pad_ids([b.input_ids for b in batch],
                 self.src_tokenizer.pad_token_id))
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id).to(torch.long)
        segment_ids = torch.LongTensor(self.pad_ids([b.segment_ids for b in batch], 0))

        state_position_ids = torch.LongTensor([b.state_position_ids for b in batch])

        op_ids = torch.LongTensor([b.op_ids for b in batch])
        domain_ids = torch.LongTensor([b.domain_id for b in batch])
        max_update = max([len(b.gen_ids) for b in batch])
        if max_update != 0:
            max_value = max([len(x) for b in batch for x in b.gen_ids])
            gen_tmp = deepcopy([self.pad_ids(b.gen_ids, 0, max_length=max_value) for b in batch])
            for x in gen_tmp:
                if len(x) < max_update:
                    diff = max_update - len(x)
                    for i in range(diff):
                        x.append([0] * max_value)
            gen_ids = self.pad_id_of_matrix(
                [torch.LongTensor(x) for x in gen_tmp],
                self.trg_tokenizer.pad_token_id,
            )
        else:
            batch_size = len(batch)
            gen_ids = torch.zeros(batch_size, 0, 0, dtype=torch.long)
            max_value = max_update = 0
        guids = [b.guid for b in batch]
        return input_ids, input_masks, segment_ids, state_position_ids, op_ids,\
            domain_ids, gen_ids, max_value, max_update, guids

    def change_slot_values(self, slot_values, input_id):
        tok_slots =[]
        for k in self.slot_meta:
            value = slot_values.get(k, '[NULL]')
            tok_value = self.src_tokenizer.encode(value, add_special_tokens=False)
            tok_slot = [self.slot_token_id] + self.slot_name_tokenized[k] + [self.s_v_token_id] + tok_value
            tok_slots.extend(tok_slot)

        if len(tok_slots) >= 512:
            tok_slots = []
            for k in self.slot_meta:
                value = slot_values.get(k, '[NULL]')
                tok_value = self.src_tokenizer.encode(value, add_special_tokens=False)[:4]
                tok_slot = [self.slot_token_id] + self.slot_name_tokenized[k] + [self.s_v_token_id] + tok_value
                tok_slots.extend(tok_slot)

        sep_1, sep_2 = torch.where(input_id[0] == self.src_tokenizer.sep_token_id)[0].tolist()
        cur_tok_history = input_id[0, 1:sep_1].tolist()
        cur_tok_users = input_id[0, sep_1+1:sep_2].tolist()
        cur_tok_slots = input_id[0, sep_2+1:].tolist()

        max_seq_len = self.max_seq - 3

        slots_len = min(len(tok_slots), max_seq_len)
        current_len = min(max_seq_len - slots_len, len(cur_tok_users))
        history_len = min(max_seq_len - slots_len - current_len, len(cur_tok_history))
        
        history_start = len(cur_tok_history) - history_len
        tok_history = [self.src_tokenizer.cls_token_id] + cur_tok_history[history_start:] + \
                [self.src_tokenizer.sep_token_id]

        current_start = len(cur_tok_users) - current_len
        tok_current = cur_tok_users[current_start:] + [self.src_tokenizer.sep_token_id]
        tok_slots = tok_slots[:slots_len]

        input_id = tok_history + tok_current + tok_slots
        input_id = torch.LongTensor(input_id).unsqueeze(0)
        segment_id = [0] * len(tok_history) + [1] * len(tok_current) + [1] * len(tok_slots)
        segment_id = torch.LongTensor(segment_id).unsqueeze(0)

        state_position_id = torch.where(input_id[0] == self.slot_token_id)[0].unsqueeze(0)
        input_mask = input_id.ne(self.src_tokenizer.pad_token_id).to(torch.long)

        return input_id, segment_id, state_position_id, input_mask
        


