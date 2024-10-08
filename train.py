import os
import argparse
import wandb
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from singmos.model import MOS_Predictor
from singmos.model import MOS_Loss
from singmos.dataset import MyDataset

import random

def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    parser.add_argument('--arch', type=str, required=False, default='SSLMOS_base', help='model architecture')
    parser.add_argument('--seed', type=int, required=False, default=1984, help='Seed of ramdom setting')
    args = parser.parse_args()

    set_all_random_seed(args.seed)
   
    datadir = args.datadir
    ckptdir = args.outdir + "_" + str(args.seed)
    if not os.path.exists(ckptdir):
            os.system('mkdir -p ' + ckptdir)
    my_checkpoint = args.finetune_from_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    with open(args.model_config) as f:
        model_config = yaml.load(f, Loader=yaml.Loader)
    
    # wandb config
    exp_config = {
        "architecture": f"{args.arch}",
        "train_batch_size": 4,
        "train_num_workers": 2,
        "valid_batch_size": 2,
        "valid_num_workers": 2,
        "learning_rate": 1e-4,
        "max_epoch": 50,
        "patience": 20,
        "random_seed": args.seed,
    }

    exp_config["model_config"] = model_config
    wdb_name = args.model_config.split("/")[-1].split(".")[0]
    
    wandb.init(
        project = "ICASSP2024",
        name = wdb_name,
        config = exp_config
    )
    config = wandb.config
    
    # dataset config 
    wavdir = os.path.join(datadir, 'wav')
    use_judge_id = model_config["model_param"]["use_judge_id"]
    if not use_judge_id:
        trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
        trainset = MyDataset(wavdir, trainlist)
    else:
        trainlist = os.path.join(datadir, 'sets/judge/train_mos_judge_list.txt')
        trainset = MyDataset(wavdir, trainlist, use_judge_id=True)
        # validlist = os.path.join(datadir, 'sets/judge/val_mos_judge_list.txt')
        # validset = MyDataset(wavdir, validlist, use_judge_id=True)
    validlist = os.path.join(datadir, 'sets/dev_mos_list.txt')
    validset = MyDataset(wavdir, validlist)
        
    trainloader = DataLoader(
        trainset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.train_num_workers, 
        collate_fn=trainset.collate_fn
    )
    validloader = DataLoader(
        validset, 
        batch_size=config.valid_batch_size, 
        shuffle=True, 
        num_workers=config.valid_num_workers,
        collate_fn=validset.collate_fn
    )
    n_steps_per_epoch = len(trainloader)

    net = MOS_Predictor(
        **model_config["model_param"]
    )
    net = net.to(device)
    
    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    # train model
    utt_loss_func = MOS_Loss(loss_type=model_config["loss_type"], use_margin=["use_margin"], margin=0.1)
    frame_loss_func = MOS_Loss(loss_type=model_config["loss_type"], use_margin=True, margin=0.1)
    optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)
    
    # patience: loss not decrease
    PREV_VAL_LOSS=9999999999
    orig_patience=config.patience
    # patience=orig_patience
    for epoch in range(1, config.epoch + 1):
        ### Train ###
        net.train()
        STEPS=0
        epoch_train_loss = 0.0
        for step, data in enumerate(trainloader):      
            judge_id = None
            if use_judge_id:
                *inputs, labels, filenames, judge_id = data
            else:
                *inputs, labels, filenames = data
            wav, wav_length, f0_start, f0_variation, f0_origin = inputs
            # move to device
            wav = wav.to(device)
            wav_length = wav_length.to(device)
            f0_start = f0_start.to(device)
            f0_variation = f0_variation.to(device)
            f0_origin = f0_origin.to(device)
            labels = labels.to(device)
            if judge_id is not None:
                judge_id = judge_id.to(device)

            batch = {}
            batch.update(
                audio=wav,
                audio_length=wav_length,
                f0_start=f0_start,
                f0_variation=f0_variation,
                f0_origin=f0_origin,
                judge_id=judge_id,
            )
            
            optimizer.zero_grad()
            outputs = net(**batch)

            utt_loss = utt_loss_func(outputs["utt_score"], labels, frame_level=False)
            if model_config["use_frame_loss"] is True:
                frame_loss = frame_loss_func(outputs["frame_score"], labels, frame_level=True)
            else:
                frame_loss = 0
            loss = utt_loss + frame_loss * model_config["alpha_frame"]

            loss.backward()
            optimizer.step()
            STEPS += 1
            epoch_train_loss += loss.item()
            
            metrics = {
                "train/total_loss": loss,
                "train/utt_loss": utt_loss,
                "train/frame_loss": frame_loss,
                "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
            }
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
            VALSTEPS+=1
            judge_id = None
            if use_judge_id:
                *inputs, labels, filenames, judge_id = data
            else:
                *inputs, labels, filenames = data
            wav, wav_length, f0_start, f0_variation, f0_origin = inputs
            # move to device
            wav = wav.to(device)
            wav_length = wav_length.to(device)
            f0_start = f0_start.to(device)
            f0_variation = f0_variation.to(device)
            f0_origin = f0_origin.to(device)
            labels = labels.to(device)
            if judge_id is not None:
                judge_id = judge_id.to(device)

            batch = {}
            batch.update(
                audio=wav,
                audio_length=wav_length,
                f0_start=f0_start,
                f0_variation=f0_variation,
                f0_origin=f0_origin,
                judge_id=judge_id,
            )
            outputs = net(**batch)

            utt_loss = utt_loss_func(outputs["utt_score"], labels, frame_level=False)
            frame_loss = 0
            if model_config["use_frame_loss"] is True:
                frame_loss = frame_loss_func(outputs["frame_score"], labels, frame_level=True)
            loss = utt_loss + frame_loss * model_config["alpha_frame"]
            epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / VALSTEPS
        val_metrics = {
            "val/epoch": epoch,
            "val/utt_loss": utt_loss,
            "val/frame_loss": frame_loss,
            "val/val_loss": avg_val_loss,
        }
        wandb.log(val_metrics)
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            PATH = os.path.join(ckptdir, 'ckpt_' + str(epoch))
            torch.save(net.state_dict(), PATH)
            orig_patience = config.patience
        else:
            orig_patience -= 1
            if orig_patience == 0:
                break
        # if epoch == config.epoch:
        #     PATH = os.path.join(ckptdir, f'ckpt_latest_{epoch}')
        #     torch.save(net.state_dict(), PATH)
    
    wandb.finish()
    print('Finished Training')

    
if __name__ == "__main__":
    main()