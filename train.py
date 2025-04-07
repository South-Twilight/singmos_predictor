import os
import argparse
import wandb
import yaml
import json
import shutil
import glob
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from singmos.model import MOS_Predictor
from singmos.model import MOS_Loss
from singmos.dataset import MyDataset

import random

def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--dataname', type=str, required=True, help='Nmae of dataset')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config')
    parser.add_argument('--finetune_from_ckpt', type=str, required=False, default=None, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    parser.add_argument('--arch', type=str, required=False, default='sslmos', help='model architecture')
    parser.add_argument('--seed', type=int, required=False, default=1234, help='Seed of ramdom setting')
    return parser


def main():
    args = get_parser().parse_args()

    set_all_random_seed(args.seed)
   
    datadir = args.datadir
    ckptdir = args.outdir
    if not os.path.exists(ckptdir):
            os.system('mkdir -p ' + ckptdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    with open(args.model_config) as f:
        model_config = yaml.load(f, Loader=yaml.Loader)
    
    # wandb config
    exp_config = {
        "model": f"{args.arch}",
        "dataset": args.dataname,
        "random_seed": args.seed,
    }
    exp_config["model_config"] = model_config

    wdb_name = args.model_config.split("/")[-1].split(".")[0] + "_" + args.dataname + "_" + str(args.seed)
    print("wdb_name: {}".format(wdb_name))
    
    wandb.init(
        project="SingMOS",
        name=wdb_name,
        config=exp_config
    )
    config = model_config
    print(config)
    with open(os.path.join(ckptdir, f"config.json"), "w") as f:
        json.dump(exp_config, f, indent=4)
    
    # auxilary information
    use_judge_id = config["model_param"]["use_judge_id"]
    use_f0 = config["model_param"]["use_f0"] or config["model_param"]["use_f0_var"]

    # For SingMOS dataset
    with open(f"{datadir}/info/split.json", "r") as f:
        split_info = json.load(f)[args.dataname]
    with open(f"{datadir}/info/score.json", "r") as f:
        score_info = json.load(f)
    
    train_set = MyDataset(split_info["train"], score_info["utterance"], datadir,  use_judge_id, use_f0)
    valid_set = MyDataset(split_info["valid"], score_info["utterance"], datadir, use_judge_id, use_f0)
        
    trainloader = DataLoader(
        train_set, 
        batch_size=config["train_batch_size"], 
        shuffle=True, 
        num_workers=config["train_num_workers"],
        collate_fn=train_set.collate_fn
    )
    validloader = DataLoader(
        valid_set, 
        batch_size=config["valid_batch_size"],
        shuffle=True, 
        num_workers=config["valid_num_workers"],
        collate_fn=valid_set.collate_fn
    )
    n_steps_per_epoch = len(trainloader)

    net = MOS_Predictor(
        **config["model_param"]
    )
    net = net.to(device)
    
    st_ckpt = args.finetune_from_ckpt
    if st_ckpt != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(st_ckpt))
    
    optimizer=None
    if config["optim"] == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(),
            **config["optim_conf"]
        )
    else:
        raise NotImplementedError(f"Optimizer {config['optim']} not implemented")

    # train model
    PREV_VAL_LOSS=9999999999
    orig_patience=config["train_patience"]
    for epoch in range(1, config["max_epoch"] + 1):
        ### Train ###
        net.train()
        STEPS=0
        epoch_train_loss = 0.0
        for step, data in enumerate(trainloader):
            wavnames, wavs, wavs_length, judge_gt_scores, mean_gt_scores, ret_dict = data
            wavs = wavs.to(device)
            wavs_length = wavs_length.to(device)
            judge_gt_scores = judge_gt_scores.to(device)
            mean_gt_scores = mean_gt_scores.to(device)
            f0_variation = None
            f0 = None
            judge_id = None
            if "f0_variation" in ret_dict:
                f0_variation = ret_dict["f0_variation"].to(device)
            if "f0" in ret_dict:
                f0 = ret_dict["f0"].to(device)
            if "judge_id" in ret_dict:
                judge_id = ret_dict["judge_id"].to(device)

            batch = {}
            batch.update(
                audio=wavs,
                audio_length=wavs_length,
                f0_variation=f0_variation,
                f0=f0,
                judge_id=judge_id,
                mean_score=mean_gt_scores,
                judge_score=judge_gt_scores,
            )
            
            optimizer.zero_grad()
            loss, stats, ret_val = net(**batch)

            loss.backward()
            optimizer.step()
            STEPS += 1
            epoch_train_loss += loss.item()
            
            metrics = {
                "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
            }
            for key, value in stats.items():
                metrics[f"train/{key}"] = value
            wandb.log(metrics)
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(epoch_train_loss / STEPS))
        
        ### Valid ###
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        VALSTEPS=0
        epoch_val_loss = 0.0
        
        for step, data in enumerate(validloader):      
            VALSTEPS += 1
            wavnames, wavs, wavs_length, judge_gt_scores, mean_gt_scores, ret_dict = data
            wavs = wavs.to(device)
            wavs_length = wavs_length.to(device)
            judge_gt_scores = judge_gt_scores.to(device)
            mean_gt_scores = mean_gt_scores.to(device)
            f0_variation = None
            f0 = None
            judge_id = None
            if "f0_variation" in ret_dict:
                f0_variation = ret_dict["f0_variation"].to(device)
            if "f0" in ret_dict:
                f0 = ret_dict["f0"].to(device)
            if "judge_id" in ret_dict:
                judge_id = ret_dict["judge_id"].to(device)

            batch = {}
            batch.update(
                audio=wavs,
                audio_length=wavs_length,
                f0_variation=f0_variation,
                f0=f0,
                judge_id=judge_id,
                mean_score=mean_gt_scores,
                judge_score=judge_gt_scores,
            )
            loss, stats, ret_val = net(**batch)

            STEPS += 1
            epoch_val_loss += loss.item()
            
            metrics = {
                "valid/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
            }
            for key, value in stats.items():
                metrics[f"valid/{key}"] = value
            wandb.log(metrics)

        avg_val_loss = epoch_val_loss / VALSTEPS
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            save_checkpoint(net, epoch, ckptdir)
            orig_patience = config["train_patience"]
        else:
            orig_patience -= 1
            if orig_patience == 0:
                break
        # if epoch == config.epoch:
        #     PATH = os.path.join(ckptdir, f'ckpt_latest_{epoch}')
        #     torch.save(net.state_dict(), PATH)
    
    wandb.finish()
    print('Finished Training')


def save_checkpoint(model, epoch, save_dir, max_keep=5):
    ckpt_name = f"ckpt_{epoch}.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    
    latest_path = os.path.join(save_dir, "valid.best.pth")
    if os.path.exists(latest_path) or os.path.islink(latest_path):
        os.remove(latest_path)  
    os.symlink(ckpt_name, latest_path)
    all_ckpts = sorted(glob.glob(os.path.join(save_dir, "ckpt_*.pth")), 
                  key=os.path.getmtime)
    if len(all_ckpts) > max_keep:
        for old_ckpt in all_ckpts[:-max_keep]:
            os.remove(old_ckpt)

    
if __name__ == "__main__":
    main()