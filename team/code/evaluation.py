import json
import argparse
from eval_utils import DSTEvaluator
from data_utils import split_slot

SLOT_META_PATH = 'data/train_dataset/slot_meta.json'


def _evaluation(preds, labels, slot_meta):
    evaluator = DSTEvaluator(slot_meta)

    evaluator.init()
    assert len(preds) == len(labels)
    
    wrong_list=[]
    correct_list=[]
    guid_compare_dict=dict()
    
    last_guid_name=""
    for k, l in labels.items():
        # k는 guid, l은 dialogue state
        p = preds.get(k)
        guid_k=k.split('-')
        main_guid=''.join(guid_k[:-1])
        turn_guid=guid_k[-1]
        if p is None:
            raise Exception(f"{k} is not in the predictions!")
        evaluator.update(l, p)
        #input_dict initiation
        if last_guid_name != main_guid:
            input_dict=dict()
            last_guid_name=main_guid
            
        if "시당" in k or "지하철" in k or "택시" in k and set(l)!=set(p):
            #맞춘 부분을 제외한 모든 dom-slot-value
            
            turn_k_dict=dict()
            for val in ((set(l) | set(p)) - (set(l) & set(p))):
                dom_slot,value=split_slot(val, get_domain_slot=True)
                #모델이 gt의 slot을 무시했을 경우 
                if dom_slot in ''.join(set(l)-set(p)) and dom_slot not in ''.join(set(p)-set(l)):
                    gt=value
                    pr='none1'
                #모델이 gt에 없는 이상한 slot을 예측하여 value를 넣은 경우
                elif dom_slot in ''.join(set(p)-set(l)) and dom_slot not in ''.join(set(l)-set(p)):
                    pr=value
                    gt='none2'
                #모델이 gt에 존재하는 slot의 value를 잘못 예측했을 경우
                else :
                    pr=value
                    for l_val in set(l):
                        _,value_gt=split_slot(l_val, get_domain_slot=True)
                        if dom_slot in l_val:
                            gt=value_gt
                            break
                            
                if gt==pr:
                    continue
                turn_k_dict[dom_slot]=(gt,pr)
            if len(turn_k_dict)!=0 :
                input_dict[turn_guid]=turn_k_dict
        wrong_list.extend(set(l)-set(p))
        correct_list.extend(set(l))
        if len(input_dict)!=0:
            guid_compare_dict[main_guid]=input_dict
        
    result = evaluator.compute()
    # print(result)
    return result,wrong_list,correct_list,guid_compare_dict


def evaluation(gt_path, pred_path):
    slot_meta = json.load(open(SLOT_META_PATH))
    gts = json.load(open(gt_path))
    preds = json.load(open(pred_path))
    eval_result,_,_ = _evaluation(preds, gts, slot_meta)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    args = parser.parse_args()
    eval_result = evaluation(args.gt_path, args.pred_path)
