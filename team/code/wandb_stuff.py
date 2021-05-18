import wandb
from attrdict import AttrDict

def setup(conf, args):
    wandb_conf = AttrDict(conf['wandb'])
    if conf['wandb']['using']:
        
        wandb.init(project=wandb_conf.project, entity=wandb_conf.entity,
                 config=args, tags=wandb_conf.tags)
        args = wandb.config
    else:
        args = AttrDict(vars(args))
    args.wandb = wandb_conf

    return args

def watch_model(args, model):
    if not args.wandb['using']:
        return
    
    wandb.watch(model)

def train_log(args, loss_dict, step, epoch):
    if not args.wandb['using']:
        return 

    if len(loss_dict) > 5:
        loss_dict = {f'train_loss': loss_dict['loss']}
    else:
        loss_dict = {f'train_{k}':v for k, v in loss_dict.items()}
    loss_dict['epoch'] = epoch
    wandb.log(loss_dict, step=step)

def val_log(args, eval_result, loss_dict, step, epoch):
    if not args.wandb['using']:
        return 

    eval_result = {f'val_{k}':v for k, v in eval_result.items()}
    eval_result['epoch'] = epoch
    wandb.log(eval_result, step=step)

    if loss_dict is not None:
        if len(loss_dict) > 5:
            loss_dict = {f'val_loss': loss_dict['loss']}
        else:
            loss_dict = {f'val_{k}':v for k, v in loss_dict.items()}
        loss_dict['epoch'] = epoch
        wandb.log(loss_dict, step=step)

def best_jga_log(args, best_score):
    if not args.wandb['using']:
        return 

    wandb.run.summary['best_val_jga'] = best_score
    