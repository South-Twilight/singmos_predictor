import os
import argparse
import wandb
import yaml
import json
import logging
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

from singmos.model import MOS_Predictor
from singmos.model import MOS_Loss
from singmos.mos_dataset import setup_dataloader_from_DATA

import random
import sys
from datetime import datetime

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
    parser.add_argument('--resume_train', type=bool, default=True, help='Resume training from latest checkpoint in outdir')
    return parser


def save_checkpoint(model, optimizer, epoch, save_dir, tag=None, max_keep=5):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = f"ckpt_{epoch}.pth" if tag is None else f"ckpt_{tag}.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)
    
    # 普通检查点只保存模型参数
    torch.save(model.state_dict(), ckpt_path)

    # latest.pth使用软链接指向最新的检查点
    latest_path = os.path.join(save_dir, "latest.pth")
    if os.path.exists(latest_path) or os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(ckpt_name, latest_path)

    # 清理多余的检查点
    all_ckpts = sorted(glob.glob(os.path.join(save_dir, "ckpt_*.pth")), key=os.path.getmtime)
    if len(all_ckpts) > max_keep:
        for old_ckpt in all_ckpts[:-max_keep]:
            os.remove(old_ckpt)


def find_latest_checkpoint(outdir):
    """查找outdir中最新的检查点文件，优先使用latest_checkpoint.pth"""
    if not os.path.exists(outdir):
        return None, 0
    
    # 首先检查latest_checkpoint.pth
    latest_checkpoint_path = os.path.join(outdir, "latest_checkpoint.pth")
    if os.path.isfile(latest_checkpoint_path):
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                return latest_checkpoint_path, checkpoint['epoch']
        except Exception as e:
            logging.warning(f"Failed to load latest_checkpoint.pth: {e}")
    
    return None, 0
    

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载检查点，返回epoch号"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查是否是latest_checkpoint.pth（包含完整状态）还是普通检查点（只有模型参数）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # latest_checkpoint.pth格式：包含完整状态
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        logging.info(f"Loaded full checkpoint from {checkpoint_path}, epoch: {epoch}")
    else:
        # 普通检查点格式：只有模型参数
        model.load_state_dict(checkpoint)
        epoch = 0  # 普通检查点没有epoch信息
        logging.info(f"Loaded model parameters from {checkpoint_path}")
    
    return epoch


def plot_loss_curve(epochs, losses, save_path):
    """绘制损失曲线并保存"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Loss curve saved to {save_path}")


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


def _setup_file_logging(outdir: str):
    """在 outdir/log 下建立文件日志，并记录当前命令行。"""
    log_dir = os.path.join(outdir, "log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join(log_dir, f"train_{timestamp}.log")

    # 增加文件 Handler，保留已有的控制台输出
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(logfile, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 记录命令行到独立文件与日志
    cmd_path = os.path.join(log_dir, f"command_{timestamp}.txt")
    cmd = " ".join(sys.argv)
    with open(cmd_path, 'w', encoding='utf-8') as f:
        f.write(cmd + "\n")
    logging.info(f"CMD: {cmd}")
    logging.info(f"Log file: {logfile}")
    logging.info(f"Command file: {cmd_path}")

    return logfile


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

    # 初始化文件日志（需在知道 outdir 后尽早调用）
    os.makedirs(args.outdir, exist_ok=True)
    _setup_file_logging(args.outdir)
    logging.info(f"Args: {vars(args)}")

    exp_config = init_wandb(args, config)
    save_experiment_config(exp_config, args.outdir)

    # 初始化损失记录
    loss_history = []

    # Data
    trainloader = setup_dataloader_from_DATA(
        config,
        args.datadir,
        train_datasets=config.get("train_datasets", ["singeval_p1"]),
        merge_diff_train=True,
    )
    logging.info(f"Train datasets: {config.get('train_datasets', ['singeval_p1'])}")
    logging.info(f"Steps per epoch: {len(trainloader)}")

    # Model & Optimizer
    net, optimizer = build_model_and_optimizer(config, device)

    # 处理resume_train或finetune_from_ckpt
    start_epoch = 1
    logging.info(f"finetune_from_ckpt: {args.finetune_from_ckpt}")
    if args.finetune_from_ckpt is not None:  
        if not os.path.exists(args.finetune_from_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.finetune_from_ckpt}")
        # 对于finetune，只加载模型权重，不加载优化器状态和epoch
        net.load_state_dict(torch.load(args.finetune_from_ckpt, map_location=device))
        logging.info(f"Loaded checkpoint from {args.finetune_from_ckpt} for finetuning")
    elif args.resume_train:
        latest_ckpt, resume_epoch = find_latest_checkpoint(args.outdir)
        if latest_ckpt is not None:
            start_epoch = load_checkpoint(net, optimizer, latest_ckpt, device) + 1
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            logging.info("No checkpoint found in outdir, starting from epoch 1")

    scaler = GradScaler(enabled=use_amp)

    n_steps_per_epoch = len(trainloader)
    best_utt_loss = float('inf')
    patience = patience_cfg

    try: 
        for epoch in range(start_epoch, max_epoch + 1):
            net.train()
            epoch_train_loss = 0.0
            logging.info(f"===== Epoch {epoch} / {max_epoch} =====")
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
                # 控制台进度条，日志中每若干步记录一次
                if (step + 1) % max(1, n_steps_per_epoch // 10) == 0 or (step + 1) == n_steps_per_epoch:
                    logging.info(f"Epoch {epoch} Step {step + 1}/{n_steps_per_epoch} loss: {metrics.get('train/loss', float(loss.item()))}")
                pbar.set_postfix({"loss": metrics.get("train/loss", 0.0)})

            avg_train_loss = epoch_train_loss / max(1, len(trainloader))
            loss_history.append(avg_train_loss)
            
            logging.info('EPOCH: ' + str(epoch))
            logging.info('AVG EPOCH TRAIN LOSS: ' + str(avg_train_loss))
            
            # 简易早停：基于训练损失/utt_loss
            if avg_train_loss < best_utt_loss:
                logging.info('Loss has decreased')
                best_utt_loss = avg_train_loss
                save_checkpoint(net, optimizer, epoch, args.outdir)
                patience = patience_cfg
            else:
                patience -= 1
                if patience <= 0:
                    logging.info('Early stopping')
                    break
    except Exception as e:
        logging.info(f"Error: {e}")
        raise e
    finally:
        # 在训练中断时保存最新的训练状态
        try:
            # 获取当前epoch（如果训练已经开始）
            current_epoch = start_epoch + len(loss_history) - 1 if loss_history else start_epoch - 1
            if current_epoch >= start_epoch:
                logging.info(f"Saving latest checkpoint at epoch {current_epoch} due to training interruption")
                # 只保存latest_checkpoint.pth，不保存普通检查点
                latest_checkpoint_path = os.path.join(args.outdir, "latest_checkpoint.pth")
                latest_checkpoint = {
                    'epoch': current_epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(latest_checkpoint, latest_checkpoint_path)
                logging.info(f"Saved latest checkpoint to {latest_checkpoint_path}")
        except Exception as save_error:
            logging.warning(f"Failed to save checkpoint on interruption: {save_error}")
        
        # 绘制损失曲线
        if args.use_wandb is False and loss_history:
            epochs = list(range(start_epoch, start_epoch + len(loss_history)))
            plot_path = os.path.join(args.outdir, 'loss_curve.png')
            plot_loss_curve(epochs, loss_history, plot_path)
        else:
            wandb.finish()
    logging.info('Finished Training')


if __name__ == "__main__":
    main()