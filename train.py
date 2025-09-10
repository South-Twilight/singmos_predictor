import os
import argparse
import wandb
import yaml
import json
import logging
import glob
from tqdm import tqdm

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

from singmos.model import MOS_Predictor
from singmos.model import MOS_Loss
from singmos.mos_dataset import setup_dataloader_from_DATA

import random

# Configure logging once, early
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
)


def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--dataname', type=str, required=True, help='Name of dataset')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config')
    parser.add_argument('--finetune_from_ckpt', type=str, default=None, help='Checkpoint path to finetune from')
    parser.add_argument('--outdir', type=str, default='checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--arch', type=str, default='sslmos', help='model architecture')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Use wandb')
    return parser
    

def init_wandb(args, config):
    exp_config = {
        "model": f"{args.arch}",
        "dataset": args.dataname,
        "random_seed": args.seed,
        "model_config": config,
    }
    if args.use_wandb:
        wdb_name = args.model_config.split("/")[-1].split(".")[0] + "_" + args.dataname + "_" + str(args.seed)
        logging.info("wandb_name: {}".format(wdb_name))
        wandb.init(project="SingEval", name=wdb_name, config=exp_config)
    else:
        logging.info("wandb is not used")
    return exp_config


def save_experiment_config(exp_config, ckptdir):
    os.makedirs(ckptdir, exist_ok=True)
    with open(os.path.join(ckptdir, f"config.json"), "w") as f:
        json.dump(exp_config, f, indent=4)


def build_model_and_optimizer(config, device):
    net = MOS_Predictor(
        **config["model_param"],
        **config["loss"],
    ).to(device)

    optim_name = config.get("optim", "sgd").lower()
    optim_conf = config.get("optim_conf", {})
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), **optim_conf)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), **optim_conf)
    else:
        raise NotImplementedError(f"Optimizer {optim_name} not implemented")

    return net, optimizer


def extract_batch(data, device):
    wavnames, wavs, wavs_length, gt_utt_scores, gt_frame_scores, ret_dict = data
    batch = {
        "audio": wavs.to(device),
        "audio_length": wavs_length.to(device),
        "gt_utt_score": gt_utt_scores.squeeze(-1).to(device),  # [B]
        "gt_frame_score": gt_frame_scores.squeeze(-1).to(device),  # [B]
        # frame_score 使用 mean_score 来对齐帧级损失（模型内部会扩展到帧数）
        "is_train": True,
    }
    # 对接新的 pitch 与 id 特征
    if "pitch_var" in ret_dict:
        batch["pitch_var"] = ret_dict["pitch_var"].to(device)
    if "pitch_note" in ret_dict:
        batch["pitch_note"] = ret_dict["pitch_note"].to(device)
    if "pitch_histogram" in ret_dict:
        batch["pitch_histogram"] = ret_dict["pitch_histogram"].to(device)
    if "pitch" in ret_dict:
        batch["pitch"] = ret_dict["pitch"].to(device)
    if "judge_id" in ret_dict:
        batch["judge_id"] = ret_dict["judge_id"].to(device)
    if "domain_id" in ret_dict:
        batch["domain_id"] = ret_dict["domain_id"].to(device)
    return batch


def save_checkpoint(model, epoch, save_dir, tag=None, max_keep=5):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = f"ckpt_{epoch}.pth" if tag is None else f"ckpt_{tag}.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)

    latest_path = os.path.join(save_dir, "latest.pth")
    # 使用软链接指向最新/最佳模型
    if os.path.exists(latest_path) or os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(ckpt_name, latest_path)

    # 清理多余的检查点
    all_ckpts = sorted(glob.glob(os.path.join(save_dir, "ckpt_*.pth")), key=os.path.getmtime)
    if len(all_ckpts) > max_keep:
        for old_ckpt in all_ckpts[:-max_keep]:
            os.remove(old_ckpt)


def main():
    args = get_parser().parse_args()

    set_all_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('DEVICE: ' + str(device))

    with open(args.model_config) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Prefer YAML training params; CLI can override when provided
    use_amp = bool(config.get("use_amp", False))
    grad_clip = float(config.get("grad_clip", 0.0))
    max_epoch = int(config.get("max_epoch", 1))
    patience_cfg = int(config.get("train_patience", 5))

    exp_config = init_wandb(args, config)
    save_experiment_config(exp_config, args.outdir)

    # Data
    trainloader = setup_dataloader_from_DATA(
        config,
        args.datadir,
        train_datasets=config.get("train_datasets", ["singeval_p1"]),
        merge_diff_train=True,
    )

    # Model & Optimizer
    net, optimizer = build_model_and_optimizer(config, device)

    # Finetune from checkpoint
    if args.finetune_from_ckpt is not None and os.path.isfile(args.finetune_from_ckpt):  
        net.load_state_dict(torch.load(args.finetune_from_ckpt, map_location=device))
        logging.info(f"Loaded checkpoint from {args.finetune_from_ckpt}")

    scaler = GradScaler(enabled=use_amp)

    n_steps_per_epoch = len(trainloader)
    best_utt_loss = float('inf')
    patience = patience_cfg

    for epoch in range(1, max_epoch + 1):
        net.train()
        epoch_train_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Train {epoch}")
        for step, data in enumerate(pbar):
            batch = extract_batch(data, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss, stats, ret_val = net(**batch)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()

            epoch_train_loss += float(loss.item())

            # Logging
            metrics = {"train/epoch": (step + 1 + (n_steps_per_epoch * (epoch - 1))) / n_steps_per_epoch}
            for key, value in stats.items():
                metrics[f"train/{key}"] = float(value.detach().item() if torch.is_tensor(value) else value)
            if args.use_wandb:
                wandb.log(metrics)
            pbar.set_postfix({"loss": metrics.get("train/loss", 0.0)})

        avg_train_loss = epoch_train_loss / max(1, len(trainloader))
        logging.info('EPOCH: ' + str(epoch))
        logging.info('AVG EPOCH TRAIN LOSS: ' + str(avg_train_loss))

        # 简易早停：基于训练损失/utt_loss
        current_utt_loss = metrics.get("train/utt_loss", avg_train_loss)
        if current_utt_loss < best_utt_loss:
            logging.info('Loss has decreased')
            best_utt_loss = current_utt_loss
            save_checkpoint(net, epoch, args.outdir)
            patience = patience_cfg
        else:
            patience -= 1
            if patience <= 0:
                logging.info('Early stopping')
                break

    wandb.finish()
    logging.info('Finished Training')


if __name__ == "__main__":
    main()