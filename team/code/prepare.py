from attrdict import AttrDict
from importlib import import_module
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import os
from tqdm.auto import tqdm
import torch
import pickle
from transformers import AutoTokenizer
from transformers import BertPreTrainedModel, BertModel, BertConfig
from data_utils import (
    load_dataset, 
    get_examples_from_dialogues, 
    convert_state_dict, 
    DSTInputExample, 
    OpenVocabDSTFeature, 
    DSTPreprocessor, 
    WOSDataset)

def set_directory(new_dir):
    global directory
    directory= new_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

gen_slot_meta = set(['관광-이름', '숙소-예약 기간', '숙소-예약 명수', '숙소-이름', '식당-예약 명수', '식당-이름', '택시-도착지', '택시-출발지', '지하철-도착지', '지하철-출발지'])
time_slot_meta = set(['식당-예약 시간', '지하철-출발 시간', '택시-도착 시간', '택시-출발 시간'])

def get_active_slot_meta(args, slot_meta):
    if args.use_domain_slot == 'gen':
        filter_slot_meta = gen_slot_meta
    elif args.use_domain_slot == 'time':
        filter_slot_meta = time_slot_meta
    elif args.use_domain_slot == 'cat':
        filter_slot_meta = set(slot_meta) - gen_slot_meta
    else:
        raise NotImplementedError(f'not implemented {args.use_domain_slot}')
    filter_domain = set(s.split('-')[0] for s in filter_slot_meta)
    return filter_slot_meta, filter_domain

def filter_inference(args, data, slot_meta, ontology):
    if args.use_domain_slot == 'basic':
        return data, slot_meta, ontology

    filter_slot_meta, filter_domain = get_active_slot_meta(args, slot_meta)
    print(f'Inferencing with only {" ".join(filter_slot_meta)}')

    old_data = data
    data = []
    for dial in old_data:
        if any([x in filter_domain for x in dial['domains']]):
            new_domains = [x for x in dial['domains'] if x in filter_domain]
            dial['domains'] = new_domains
            if len(new_domains) > 0:
                data.append(dial)

    print(f'Filtered {len(old_data)} -> {len(data)}')

    slot_meta = sorted(list(filter_slot_meta))
    new_ontology = {}
    for cur_slot_meta in slot_meta:
        new_ontology[cur_slot_meta] = ontology[cur_slot_meta]
    ontology = new_ontology

    return data, slot_meta, ontology

def get_data(args):
    print(f'using train: {args.train_file_name}')
    train_data_file = f"{args.data_dir}/{args.train_file_name}"
    data = json.load(open(train_data_file))
    
    if 'train_from_trained' not in args:
        args.train_from_trained = None
    if args.train_from_trained is None:
        slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
        ontology = json.load(open(args.ontology_root))
    else:
        slot_meta = json.load(open(f"{args.train_from_trained}/slot_meta.json"))
        ontology = json.load(open(f'{args.train_from_trained}/edit_ontology_metro.json'))

    if args.use_convert_ont:
        if args.convert_time != 'none':
            convert_time_dict = getattr(import_module('change_ont_value'), args.convert_time)
            print(f'Change Time Format: xx:xx -> {convert_time_dict.example}')
            print(f'Change {"  ".join(convert_time_dict.applied)}')
            for cat in convert_time_dict.applied:
                if cat in ontology:
                    ontology[cat] = [convert_time_dict.convert(x) for x in ontology[cat]]
            args.convert_time_dict = convert_time_dict
        else:
            args.convert_time_dict = None
            
    if args.use_domain_slot == 'basic':
        if args.use_small_data:
            data = data[:100]
        return data, slot_meta, ontology

    filter_slot_meta, filter_domain = get_active_slot_meta(args, slot_meta)

    print(f'Using only {" ".join(filter_slot_meta)}')
    old_data = data
    data = []
    for dial in old_data:
        if any([x in filter_domain for x in dial['domains']]):
            new_domains = [x for x in dial['domains'] if x in filter_domain]
            dial['domains'] = new_domains
            new_dialogue = []
            is_worthy = False
            for x in dial['dialogue']:
                if 'state' in x:
                    new_state = [st for st in x['state'] if '-'.join(st.split('-')[:2])  in filter_slot_meta]
                    x['state'] = new_state
                    if not is_worthy and len(new_state) > 0:
                        is_worthy = True
                new_dialogue.append(x)
            dial['dialogue'] = new_dialogue

            if is_worthy:
                data.append(dial)
    print(f'Filtered {len(old_data)} -> {len(data)}')
    slot_meta = sorted(list(filter_slot_meta))
    new_ontology = {}
    for cur_slot_meta in slot_meta:
        new_ontology[cur_slot_meta] = ontology[cur_slot_meta]
    ontology = new_ontology

    if args.use_small_data:
        data = data[:100]

    return data, slot_meta, ontology

def convert_train_val_features(processor, train_examples, dev_examples):
    if train_examples is not None:
        train_features = processor.convert_examples_to_features(train_examples, which='train')
    else:
        train_features = None
    if dev_examples is not None:
        dev_features = processor.convert_examples_to_features(dev_examples, which='val')
    else:
        dev_features = None
    return train_features, dev_features

def get_stuff(args, train_data, dev_data, slot_meta, ontology):
    if args.preprocessor == 'TRADEPreprocessor':
        user_first = False
        dialogue_level = False
        processor_kwargs = AttrDict(
            use_zero_segment_id=args.use_zero_segment_id,
        )
    elif args.preprocessor == 'SUMBTPreprocessor':
        user_first = True
        dialogue_level = True
        max_turn = max([len(e['dialogue']) for e in train_data])
        processor_kwargs = AttrDict(
            ontology=ontology,
            max_turn_length=max_turn,
            max_seq_length=args.max_seq_length,
            model_name_or_path=args.model_name_or_path,
            args=args,
        )
    elif args.preprocessor == 'SOMDSTPreprocessor':
        user_first = False
        dialogue_level = False
        processor_kwargs = AttrDict()
    else:
        raise NotImplementedError()

    if train_data is not None:
        train_examples = get_examples_from_dialogues(
            train_data, user_first=user_first, use_sys_usr_sys=args.use_sys_usr_sys_turn,
                dialogue_level=dialogue_level, which='train'
        )
    else:
        train_examples = None

    if dev_data is not None:
        dev_examples = get_examples_from_dialogues(
            dev_data, user_first=user_first, use_sys_usr_sys=args.use_sys_usr_sys_turn,
            dialogue_level=dialogue_level, which='val'
        )
    else:
        dev_examples = None

    # Define Preprocessor
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = getattr(import_module('preprocessor'), args.preprocessor)(
        slot_meta, tokenizer, **processor_kwargs
    )

    if args.preprocessor == 'TRADEPreprocessor':
        args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr

    # Extracting Featrues
    # print('Converting examples to features')
    if args.use_cache_examples2features:
        postfix = '_small' if args.use_small_data else ''
        if not os.path.exists(f'{args.data_dir}/cache'):
            os.mkdir(f'{args.data_dir}/cache')
        if args.refresh_cache or not os.path.exists(f'{args.data_dir}/cache/{args.preprocessor}/train{postfix}.pkl'):
            print('Saving to Cache')
            train_features, dev_features = convert_train_val_features(processor, 
                train_examples, dev_examples)

            if not os.path.exists(f'{args.data_dir}/cache/{args.preprocessor}'):
                os.mkdir(f'{args.data_dir}/cache/{args.preprocessor}')

            pickle.dump(train_features, open(f'{args.data_dir}/cache/{args.preprocessor}/train{postfix}.pkl', 'wb'))
            pickle.dump(dev_features, open(f'{args.data_dir}/cache/{args.preprocessor}/dev{postfix}.pkl', 'wb'))
        else:
            print('Loaded from Cache')
            train_features = pickle.load(open(f'{args.data_dir}/cache/{args.preprocessor}/train{postfix}.pkl', 'rb'))
            dev_features = pickle.load(open(f'{args.data_dir}/cache/{args.preprocessor}/dev{postfix}.pkl', 'rb'))

    else:
        train_features, dev_features = convert_train_val_features(processor, 
                train_examples, dev_examples)
    
    return tokenizer, processor, train_features, dev_features


def tokenize_ontology(ontology, tokenizer, max_seq_length):
    slot_types = []
    slot_values = []
    for k, v in ontology.items():
        tokens = tokenizer.encode(k)
        if len(tokens) < max_seq_length:
            gap = max_seq_length - len(tokens)
            tokens.extend([tokenizer.pad_token_id] * gap)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        slot_types.append(tokens)
        slot_value = []
        for vv in v:
            tokens = tokenizer.encode(vv)
            if len(tokens) < max_seq_length:
                gap = max_seq_length - len(tokens)
                tokens.extend([tokenizer.pad_token_id] * gap)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            slot_value.append(tokens)
        slot_values.append(torch.LongTensor(slot_value))
    return torch.LongTensor(slot_types), slot_values

def get_model(args, tokenizer, ontology, slot_meta):
    if args.ModelName == 'TRADE':
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )

        model_kwargs = AttrDict(
            slot_meta=tokenized_slot_meta
        )
        from_pretrained=False
    elif args.ModelName == 'SUMBT':
        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, args.max_label_length)
        num_labels = [len(s) for s in slot_values_ids]

        if 'use_no_lookup' not in args:
            args.use_no_lookup = False
        model_kwargs = AttrDict(
            num_labels=num_labels,
            device=args.device,
            use_no_lookup=args.use_no_lookup,
        )
        from_pretrained=False
    elif args.ModelName == 'SOM_DST':
        model_kwargs = AttrDict(
            n_op=4,
            n_domain=5,
            update_id=1,
            len_tokenizer=len(tokenizer),
            slot_token_id=tokenizer.encode('[SLOT]', add_special_tokens=False)[0],
        )
        from_pretrained=True
    else:
        raise NotImplementedError()

    pbar = tqdm(desc=f'Making {args.model_class} model -- waiting...', bar_format='{desc} -> {elapsed}')
    if from_pretrained:
        model_config = BertConfig.from_pretrained(args.model_name_or_path)
        # model_config.dropout = args.dropout
        # model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
        # model_config.hidden_dropout_prob = args.hidden_dropout_prob

        model = getattr(import_module('model'), args.model_class)(model_config, **model_kwargs)
    else:
        model = getattr(import_module('model'), args.model_class)(
            args, **model_kwargs
        )
    pbar.set_description(f'Making {args.model_class} model -- DONE')    
    pbar.close()

    # if args.ModelName == 'TRADE':
    #     pbar = tqdm(desc='Setting subword embedding -- waiting...', bar_format='{desc} -> {elapsed}')
    #     model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화    
    #     pbar.set_description('Setting subword embedding -- DONE')
    #     pbar.close()
    if args.ModelName == 'SUMBT':
        print('Initializing slot value lookup --------------')
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV        
        print('Finished initializing slot value lookup -----')
    if args.ModelName == 'SOM_DST':
        added = ['[SLOT]', '[A-U]', '[S-V]', '[NULL]', '[EOS]']
        for add_tok in added:
            add_tok_idx = tokenizer.encode(add_tok, add_special_tokens=False)[0]
            model.encoder.bert.embeddings.word_embeddings.weight.data[add_tok_idx].normal_(mean=0.0,
                     std=0.02)

    return model