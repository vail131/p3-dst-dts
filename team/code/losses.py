import torch
import torch.nn as nn
import torch.nn.functional as F
from model import masked_cross_entropy_for_value

from attrdict import AttrDict

from focal_loss.focal_loss import FocalLoss

class Trade_Loss:
    def __init__(self, pad_token_id, n_gate):
        self.loss_fnc_1 = masked_cross_entropy_for_value
        self.loss_fnc_2 = nn.CrossEntropyLoss()
        self.pad_token_id = pad_token_id
        self.n_gate = n_gate

    def __call__(self, all_point_outputs, all_gate_outputs, target_ids, gating_ids):
        loss_1 = self.loss_fnc_1(
            all_point_outputs.contiguous(),
            target_ids.contiguous().view(-1),
            self.pad_token_id
        )

        loss_2 = self.loss_fnc_2(
            all_gate_outputs.contiguous().view(-1, self.n_gate),
            gating_ids.contiguous().view(-1),
        )

        loss = loss_1 + loss_2

        return AttrDict(
            loss = loss,
            gen_loss = loss_1,
            gating_loss = loss_2,
        )

class SUBMT_Loss:
    def __init__(self):
        self.nll = nn.CrossEntropyLoss(ignore_index=-1)

    def __call__(self, outputs, labels):
        ds, ts, js = labels.shape
        loss_slot = []
        loss = 0
        for i in range(js):
            _dist = outputs[i]
            label = labels[:, :, i]
            _loss = self.nll(_dist.view(ds * ts, -1), label.view(-1))
            loss_slot.append(_loss)
            loss += _loss

        ret = AttrDict(loss=loss)
        for i, v in enumerate(loss_slot):
            ret[f'loss_slot_{i}'] = v
        return ret
        
def focal_loss(input_tensor, target_tensor, alpha=0.25, gamma=2):
    minus_log_pt = F.cross_entropy(input_tensor, target_tensor, reduction='none')
    pt = torch.exp(-minus_log_pt)
    return (alpha* (1-pt)**gamma * minus_log_pt).mean()

class SOM_DST_Loss:
    def __init__(self, pad_token_id, exclude_domain=False, device=torch.device('cuda')):
        self.loss_fnc_g= masked_cross_entropy_for_value
        # weight = torch.FloatTensor([1, 1, 45, 1]).to(device)
        self.loss_fnc = nn.CrossEntropyLoss()
        self.domain_loss_func = nn.CrossEntropyLoss()
        self.pad_token_id = pad_token_id
        self.exclude_domain = exclude_domain

    def __call__(self, domain_scores, state_scores, gen_scores,
            domain_ids, op_ids, gen_ids):
        loss_s = self.loss_fnc(state_scores.view(-1, 4), op_ids.view(-1))
        return AttrDict(
            loss_state=loss_s,
            loss=loss_s
        )
        if gen_scores.size(1) == 0:
            loss_g = torch.zeros(1).to(gen_scores.device)
        else:
            loss_g = self.loss_fnc_g(gen_scores.contiguous(), gen_ids.contiguous(), self.pad_token_id)
        loss = loss_s + loss_g

        ret = AttrDict(
            loss_state=loss_s,
            loss_gen=loss_g,
        )
        if self.exclude_domain is not True:
            loss_d = self.domain_loss_func(domain_scores.view(-1, 5), domain_ids.view(-1))
            loss = loss + loss_d
            ret['loss_domain'] = loss_d

        ret['loss'] = loss
        return ret