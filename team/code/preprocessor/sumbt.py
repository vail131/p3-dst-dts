from data_utils import convert_state_dict
from data_utils import OntologyDSTFeature, DSTPreprocessor, _truncate_seq_pair
from importlib import import_module
from tqdm.auto import tqdm
import torch

class SUMBTPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=64,
        max_turn_length=14,
        model_name_or_path=None,
        args=None,
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.max_seq_length = max_seq_length
        self.max_turn_length = max_turn_length
        self.model_name_or_path = model_name_or_path

        self.use_convert_ont = args.use_convert_ont
        if args.use_convert_ont:
            self.convert_time_dict = args.convert_time_dict
        
            if isinstance(self.convert_time_dict['convert'], str):
                self.convert_time_dict['convert'] = getattr(import_module('change_ont_value'),
                    self.convert_time_dict['convert'].split('.')[1])
                self.convert_time_dict['revert'] = getattr(import_module('change_ont_value'),
                    self.convert_time_dict['revert'].split('.')[1])

    def _convert_example_to_feature(self, example):
        guid = example[0].guid.rsplit("-", 1)[0]  # dialogue_idx
        turns = []
        token_types = []
        labels = []
        num_turn = None
        for turn in example[: self.max_turn_length]:
            assert len(turn.current_turn) == 2
            uttrs = []
            for segment_idx, uttr in enumerate(turn.current_turn):
                token = self.src_tokenizer.encode(uttr, add_special_tokens=False)
                uttrs.append(token)

            _truncate_seq_pair(uttrs[0], uttrs[1], self.max_seq_length - 3)
            tokens = (
                [self.src_tokenizer.cls_token_id]
                + uttrs[0]
                + [self.src_tokenizer.sep_token_id]
                + uttrs[1]
                + [self.src_tokenizer.sep_token_id]
            )

            if self.model_name_or_path == 'xlm-roberta-base': # roberta는 token_types 사용안함
                token_type = [0] * (len(uttrs[0]) + 2 + len(uttrs[1]) + 1)
            else:
                token_type = [0] * (len(uttrs[0]) + 2) + [1] * (len(uttrs[1]) + 1)
            if len(tokens) < self.max_seq_length:
                gap = self.max_seq_length - len(tokens)
                tokens.extend([self.src_tokenizer.pad_token_id] * gap)
                token_type.extend([0] * gap)
            turns.append(tokens)
            token_types.append(token_type)
            label = []
            if turn.label:
                slot_dict = convert_state_dict(turn.label)
            else:
                slot_dict = {}
            for slot_type in self.slot_meta:
                value = slot_dict.get(slot_type, "none")

                if self.use_convert_ont:
                    if self.convert_time_dict is not None and \
                        slot_type in self.convert_time_dict['applied']:
                        value = self.convert_time_dict['convert'](value)
                
                if self.use_convert_ont:
                    if self.convert_time_dict is not None and \
                        slot_type in self.convert_time_dict['applied']:
                        value = self.convert_time_dict['convert'](value)
                
                if value in self.ontology[slot_type]:
                    label_idx = self.ontology[slot_type].index(value)
                else:
                    label_idx = self.ontology[slot_type].index('none')
               
                label.append(label_idx)
            labels.append(label)
        num_turn = len(turns)
        if len(turns) < self.max_turn_length:
            gap = self.max_turn_length - len(turns)
            for _ in range(gap):
                dummy_turn = [self.src_tokenizer.pad_token_id] * self.max_seq_length
                dummy_token_type = [0] * self.max_seq_length
                turns.append(dummy_turn)
                token_types.append(dummy_token_type)
                dummy_label = [-1] * len(self.slot_meta)
                labels.append(dummy_label)
        return OntologyDSTFeature(
            guid=guid,
            input_ids=turns,
            segment_ids=token_types,
            num_turn=num_turn,
            target_ids=labels,
        )

    def convert_examples_to_features(self, examples, which=''):
        tdata = tqdm(examples, desc=f'Converting {which} examples to features')
        return list(map(self._convert_example_to_feature, tdata))

    def recover_state(self, pred_slots, num_turn):
        states = []
        
        for pred_slot in pred_slots[:num_turn]:
            state = []
            for s, p in zip(self.slot_meta, pred_slot):
                v = self.ontology[s][p]
                if self.use_convert_ont:
                    if self.convert_time_dict is not None and \
                        s in self.convert_time_dict['applied']:
                        v = self.convert_time_dict['revert'](v)
                if v != 'none':
                    state.append(f'{s}-{v}')
                    
            states.append(state)
            
        return states

    def collate_fn(self, batch):
        guids = [b.guid for b in batch]
        input_ids = torch.LongTensor([b.input_ids for b in batch])
        segment_ids = torch.LongTensor([b.segment_ids for b in batch])
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id)
        target_ids = torch.LongTensor([b.target_ids for b in batch])
        num_turns = [b.num_turn for b in batch]
        return input_ids, segment_ids, input_masks, target_ids, num_turns, guids