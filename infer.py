import os
import argparse
import scipy.stats

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from singmos.model import MOS_Predictor, MOS_Loss
from singmos.dataset import MyDataset

import glob
import yaml
import json

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


def infer_mos(model, dataloader, dset, device):
    pred_utt_mos = {}  # filename : prediction
    pred_sys_list = {}
    
    print(f'Starting prediction over {dset}')

    for i, data in enumerate(dataloader, 0):
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
        
        loss, stats, ret_val = model(**batch)

        utt_mos = ret_val["mean_utt_score"].cpu().detach().numpy()[0][0]
        utt_id = wavnames[0]
        sys_id = systemID(utt_id)
        pred_utt_mos[wavnames[0]] = utt_mos
        if sys_id not in pred_sys_list:
            pred_sys_list[sys_id] = []
        pred_sys_list[sys_id].append(utt_mos)

    pred_sys_mos = {}
    for k, v in pred_sys_list.items():
        avg_MOS = sum(v) / (len(v) * 1.0)
        pred_sys_mos[k] = avg_MOS

    return pred_sys_mos, pred_utt_mos


def calc_metrics(gt_utt_MOS, pred_utt_MOS, gt_sys_MOS, pred_sys_MOS, dset):
    print(f'Starting prediction over {dset}')
    metrics_answer = {}
    metrics_answer["system"] = {}
    metrics_answer["utterance"] = {}

    assert len(gt_utt_MOS) == len(pred_utt_MOS)
    assert len(gt_sys_MOS) == len(pred_sys_MOS)

    ## compute correls.
    sorted_uttIDs = sorted(gt_utt_MOS.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = gt_utt_MOS[uttID]
        p = pred_utt_MOS[uttID]
        ts.append(t)
        ps.append(p)

    truths = np.array(ts)
    preds = np.array(ps)

    ### UTTERANCE
    MSE=np.mean((truths-preds)**2)
    metrics_answer["utterance"]["MSE"] = MSE
    LCC=np.corrcoef(truths, preds)
    metrics_answer["utterance"]["LCC"] = LCC[0][1]
    SRCC=scipy.stats.spearmanr(truths.T, preds.T)
    metrics_answer["utterance"]["SRCC"] = SRCC[0]
    KTAU=scipy.stats.kendalltau(truths, preds)
    metrics_answer["utterance"]["KTAU"] = KTAU[0]
    print('[UTTERANCE] Test error= %f' % MSE)
    print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])
    print('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    ### SYSTEM
    ## make lists sorted by system
    pred_sysIDs = sorted(pred_sys_MOS.keys())
    sys_p = []
    sys_t = []
    for sysID in pred_sysIDs:
        sys_p.append(pred_sys_MOS[sysID])
        sys_t.append(gt_sys_MOS[sysID])

    sys_true = np.array(sys_t)
    sys_predicted = np.array(sys_p)

    MSE=np.mean((sys_true-sys_predicted)**2)
    metrics_answer["system"]["MSE"] = MSE
    LCC=np.corrcoef(sys_true, sys_predicted)
    metrics_answer["system"]["LCC"] = LCC[0][1]
    SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
    metrics_answer["system"]["SRCC"] = SRCC[0]
    KTAU=scipy.stats.kendalltau(sys_true, sys_predicted)
    metrics_answer["system"]["KTAU"] = KTAU[0]
    print('[SYSTEM] Test error= %f' % MSE)
    print('[SYSTEM] Linear correlation coefficient= %f' % LCC[0][1])
    print('[SYSTEM] Spearman rank correlation coefficient= %f' % SRCC[0])
    print('[SYSTEM] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    return metrics_answer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    # parser.add_argument('--dataname', type=str, required=True, help='Nmae of dataset')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--answer_dir', type=str, required=False, default='answer', help='Output directory for your answer file')
    parser.add_argument('--seed', type=int, required=False, default=1984, help='Seed of ramdom setting')
    return parser


def main():
    args = get_parser().parse_args()
    
    datadir = args.datadir
    answer_dir = args.answer_dir
    make_dir(answer_dir)

    # load model
    ckpt = args.ckpt
    print(f'infer with model: {ckpt}\n')

    with open(args.model_config) as f:
        model_config = yaml.load(f, Loader=yaml.Loader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))
   
    model = MOS_Predictor(
        **model_config["model_param"]
    ) 
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt))

    # load infer data
    with open(f"{datadir}/info/split.json", "r") as f:
        split_info = json.load(f)
    with open(f"{datadir}/info/score.json", "r") as f:
        score_info = json.load(f)

    use_f0 = model_config["model_param"]["use_f0"] or model_config["model_param"]["use_f0_var"]

    test_sets = ["singmos_v1", "singmos_v2"]
    
    for test_set in test_sets:
        # Load model
        print('Loading data')
        eval_set = MyDataset(split_info[test_set]["eval"], score_info["utterance"], datadir, use_judge_id=False, use_f0=use_f0)
        eval_loader = DataLoader(
            eval_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=eval_set.collate_fn
        )

        # Infer model
        metrics_answer = {}
        pred_sys_MOS, pred_utt_MOS  = infer_mos(model, eval_loader, test_set, device)
        gt_sys_MOS = { k: float(score_info["system"][k]) for k in eval_set.get_sys_names() }
        gt_utt_MOS = { k: float(score_info["utterance"][k]["score"]["mos"]) for k in eval_set.get_utt_names() }
        metrics_answer = calc_metrics(gt_utt_MOS, pred_utt_MOS, gt_sys_MOS, pred_sys_MOS, test_set)

        # write metrics
        make_dir(os.path.join(answer_dir, test_set, "metrics"))
        filename = args.model_config.split("/")[-1].split(".")[0] + "_" + str(args.seed)
        with open(os.path.join(answer_dir, test_set, "metrics", filename + ".json"), "w", encoding="utf-8") as f:
            json.dump(metrics_answer, f, cls=NumpyEncoder, indent=4)
        print('Metrics writen in {}'.format(os.path.join(answer_dir, test_set, "metrics", filename + ".json")))

        # write utterance mos
        make_dir(os.path.join(answer_dir, test_set, "utt_mos"))
        with open(os.path.join(answer_dir, test_set, "utt_mos", filename + ".txt"), "w", encoding="utf-8") as f:
            for k, v in pred_utt_MOS.items():
                outl = k.split('.')[0] + ',' + str(v) + '\n'
                f.write(outl)
        print('Utt MOS writen in {}'.format(os.path.join(answer_dir, test_set, "utt_mos", filename + ".txt")))

if __name__ == '__main__':
    main()
