import os
import argparse
import scipy.stats

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from singmos.model import MOS_Predictor, MOS_Loss
from singmos.mos_dataset import MOSDataset

import glob
import yaml
import json

import logging
logging.basicConfig(level=logging.INFO)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def systemID(uttID):
    return uttID.split('-')[0]


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def build_eval_loader(datadir, config, eval_list, score_info, sys_info):
    eval_set = MOSDataset(
        datadir=datadir,
        utt_list=eval_list,
        score_infos=score_info["utterance"],
        sys_info=sys_info,
        use_domain_id=config["model_param"].get("use_domain_id", False),
        use_judge_id=config["model_param"].get("use_judge_id", False),
        use_pitch=config["model_param"].get("use_pitch", False),
        pitch_type=config["model_param"].get("pitch_type", "note"),
        sample_rate=config["model_param"].get("sample_rate", 16000),
        max_duration=config["model_param"].get("max_duration", 10.0),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=15,
        shuffle=False,
        num_workers=8,
        collate_fn=eval_set.collate_fn,
    )
    return eval_loader, eval_set, score_info


def infer_mos(model, dataloader, dset, device):
    pred_utt_mos = {}
    pred_sys_list = {}

    logging.info(f'Starting prediction over {dset}')

    for i, data in enumerate(dataloader, 0):
        wavnames, wavs, wavs_length, gt_utt_scores, gt_frame_scores, ret_dict = data
        wavs = wavs.to(device)
        wavs_length = wavs_length.to(device)
        batch_size = len(wavnames)

        batch = {
            "audio": wavs,
            "audio_length": wavs_length,
            "is_train": False,
        }
        # optional features
        if "pitch_var" in ret_dict:
            batch["pitch_var"] = ret_dict["pitch_var"].to(device)
        if "pitch_note" in ret_dict:
            batch["pitch_note"] = ret_dict["pitch_note"].to(device)
        if "pitch_histogram" in ret_dict:
            batch["pitch_histogram"] = ret_dict["pitch_histogram"].to(device)
        if "pitch" in ret_dict:
            batch["pitch"] = ret_dict["pitch"].to(device)
        
        # Handle judge_id and domain_id with defaults
        if "judge_id" in ret_dict:
            batch["judge_id"] = ret_dict["judge_id"].to(device)
        elif model.use_judge_id:
            # Default judge_id = 0 when model expects it but data doesn't have it
            batch["judge_id"] = torch.zeros(batch_size, dtype=torch.long, device=device)
            
        if "domain_id" in ret_dict:
            batch["domain_id"] = ret_dict["domain_id"].to(device)
        elif model.use_domain_id:
            # Default domain_id = 1 when model expects it but data doesn't have it
            batch["domain_id"] = torch.ones(batch_size, dtype=torch.long, device=device)

        with torch.no_grad():
            loss, stats, ret_val = model(**batch)

        # Handle batch predictions
        utt_scores = ret_val["utt_score"].squeeze().detach().cpu()  # [B]
        if utt_scores.dim() == 0:  # Handle case where B=1
            utt_scores = utt_scores.unsqueeze(0)
        
        # Process each sample in the batch
        for j in range(batch_size):
            utt_mos = utt_scores[j].item()
            utt_id = wavnames[j]
            sys_id = systemID(utt_id)
            pred_utt_mos[utt_id] = utt_mos
            if sys_id not in pred_sys_list:
                pred_sys_list[sys_id] = []
            pred_sys_list[sys_id].append(utt_mos)

    pred_sys_mos = {k: (sum(v) / (len(v) * 1.0)) for k, v in pred_sys_list.items()}
    return pred_sys_mos, pred_utt_mos


def calc_metrics(gt_utt_MOS, pred_utt_MOS, gt_sys_MOS, pred_sys_MOS, dset):
    logging.info(f'Starting prediction over {dset}')
    metrics_answer = {"system": {}, "utterance": {}}

    assert len(gt_utt_MOS) == len(pred_utt_MOS)
    assert len(gt_sys_MOS) == len(pred_sys_MOS)

    # UTTERANCE
    sorted_uttIDs = sorted(gt_utt_MOS.keys())
    truths = np.array([gt_utt_MOS[u] for u in sorted_uttIDs])
    preds = np.array([pred_utt_MOS[u] for u in sorted_uttIDs])

    MSE = np.mean((truths - preds) ** 2)
    LCC = np.corrcoef(truths, preds)[0][1]
    SRCC = scipy.stats.spearmanr(truths.T, preds.T)[0]
    KTAU = scipy.stats.kendalltau(truths, preds)[0]

    metrics_answer["utterance"].update(MSE=MSE, LCC=LCC, SRCC=SRCC, KTAU=KTAU)
    logging.info('[UTTERANCE] Test error= %f' % MSE)
    logging.info('[UTTERANCE] Linear correlation coefficient= %f' % LCC)
    logging.info('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC)
    logging.info('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU)

    # SYSTEM
    pred_sysIDs = sorted(pred_sys_MOS.keys())
    sys_true = np.array([gt_sys_MOS[s] for s in pred_sysIDs])
    sys_pred = np.array([pred_sys_MOS[s] for s in pred_sysIDs])

    MSE = np.mean((sys_true - sys_pred) ** 2)
    LCC = np.corrcoef(sys_true, sys_pred)[0][1]
    SRCC = scipy.stats.spearmanr(sys_true.T, sys_pred.T)[0]
    KTAU = scipy.stats.kendalltau(sys_true, sys_pred)[0]

    metrics_answer["system"].update(MSE=MSE, LCC=LCC, SRCC=SRCC, KTAU=KTAU)
    logging.info('[SYSTEM] Test error= %f' % MSE)
    logging.info('[SYSTEM] Linear correlation coefficient= %f' % LCC)
    logging.info('[SYSTEM] Spearman rank correlation coefficient= %f' % SRCC)
    logging.info('[SYSTEM] Kendall Tau rank correlation coefficient= %f' % KTAU)

    return metrics_answer


def write_metrics(metrics_answer, pred_utt_MOS, pred_dir, config_name):
    filename = config_name.split("/")[-1].split(".")[0]
    with open(os.path.join(pred_dir, "metrics" + ".json"), "w", encoding="utf-8") as f:
        json.dump(metrics_answer, f, cls=NumpyEncoder, indent=4)
    logging.info('Metrics writen in {}'.format(os.path.join(pred_dir, "metrics" + ".json")))
    with open(os.path.join(pred_dir, "utt_mos" + ".txt"), "w", encoding="utf-8") as f:
        for k, v in pred_utt_MOS.items():
            outl = k.split('.')[0] + ',' + str(v) + '\n'
            f.write(outl)
    logging.info('Utt MOS writen in {}'.format(os.path.join(pred_dir, "utt_mos" + ".txt")))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--answer_dir', type=str, default='answer', help='Output directory for your answer file')
    parser.add_argument('--seed', type=int, default=1984, help='Seed of random setting')
    parser.add_argument('--test_sets', action='append', default=[], help='Test set names in split.json to evaluate (can be used multiple times)')
    return parser


def main():
    args = get_parser().parse_args()
    logging.info(f"test args: {args.test_sets}")

    datadir = args.datadir
    answer_dir = args.answer_dir
    make_dir(answer_dir)

    # load model
    ckpt = args.ckpt
    logging.info(f'infer with model: {ckpt}\n')

    with open(args.model_config) as f:
        model_config = yaml.load(f, Loader=yaml.Loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('DEVICE: ' + str(device))

    model = MOS_Predictor(
        **model_config["model_param"],
        **model_config["loss"],
    )
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # load split
    with open(f"{datadir}/info/split.json", "r") as f:
        split_info = json.load(f)
    with open(f"{datadir}/info/score.json", "r") as f:
        score_info = json.load(f)
    with open(f"{datadir}/info/sys_info.json", "r") as f:
        sys_info = json.load(f)

    for test_set in args.test_sets:
        logging.info(f'Loading data {test_set}')
        eval_list = split_info[test_set]["eval"]
        eval_loader, eval_set, score_info = build_eval_loader(datadir, model_config, eval_list, score_info, sys_info)

        # Infer model
        all_pred_sys_MOS, all_pred_utt_MOS = infer_mos(model, eval_loader, test_set, device)
        # 更安全的版本，处理不同的数据结构
        all_gt_sys_MOS = {}
        for k in eval_set.get_sys_names():
            sys_score = score_info["system"][k]
            all_gt_sys_MOS[k] = float(sys_score["score"])
        all_gt_utt_MOS = {k: float(score_info["utterance"][k]["score"]["mos"]) for k in eval_set.get_utt_names()}

        # All
        pred_dir = os.path.join(answer_dir, os.path.basename(args.ckpt).split(".")[0])
        make_dir(pred_dir)
        metrics_answer = calc_metrics(all_gt_utt_MOS, all_pred_utt_MOS, all_gt_sys_MOS, all_pred_sys_MOS, test_set + "-ALL")
        write_metrics(metrics_answer, all_pred_utt_MOS, pred_dir, args.model_config)

    
if __name__ == '__main__':
    main()
