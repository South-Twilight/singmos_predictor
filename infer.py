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


def systemID(uttID):
    return uttID.split('-')[0]

def find_latest_ckpt(folder_path, is_best = True):
    files = glob.glob(os.path.join(folder_path, '*'))
    if not files:
        print("The folder is empty.")
        return None

    if is_best is True:
        filtered_files = [f for f in files if "latest" not in os.path.basename(f)]
    else:
        filtered_files = files

    if not filtered_files:
        print("No suitable files found.")
        return None

    latest_file = max(filtered_files, key=os.path.getmtime)
    return latest_file


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--finetuned_checkpoint', type=str, required=True, help='Path to finetuned MOS prediction checkpoint.')
    parser.add_argument('--answer_dir', type=str, required=False, default='answer', help='Output directory for your answer file')
    parser.add_argument('--seed', type=int, required=False, default=1984, help='Seed of ramdom setting')
    args = parser.parse_args()
    
    if os.path.isdir(args.finetuned_checkpoint + "_" + str(args.seed)):
        my_checkpoint = find_latest_ckpt(args.finetuned_checkpoint + "_" + str(args.seed))
    else:
        my_checkpoint = args.finetuned_checkpoint
    print(f'infer with model: {my_checkpoint}\n')
    datadir = args.datadir
    answer_dir = args.answer_dir
    fn = args.model_config.split("/")[-1].split(".")[0] + "_" + str(args.seed)
    system_csv_path = os.path.join(datadir, 'mydata_system.csv')

    make_dir(answer_dir)

    with open(args.model_config) as f:
        model_config = yaml.load(f, Loader=yaml.Loader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))
   
    model = MOS_Predictor(
        **model_config["model_param"]
    ) 
    model.to(device)
    model.eval()

    model.load_state_dict(torch.load(my_checkpoint))


    wavdir = os.path.join(datadir, 'wav')
    evallist = os.path.join(datadir, 'sets/eval_mos_list.txt')

    print('Loading data')
    evalset = MyDataset(wavdir, evallist)
    evalloader = DataLoader(evalset, batch_size=1, shuffle=False, num_workers=2, collate_fn=evalset.collate_fn)

    total_loss = 0.0
    predictions = {}  # filename : prediction
    frame_loss_func = MOS_Loss(loss_type=model_config["loss_type"], use_margin=model_config["use_margin"], margin=0.1)
    utt_loss_func = MOS_Loss(loss_type=model_config["loss_type"], use_margin=model_config["use_margin"], margin=0.1)
    use_judge_id = model_config["model_param"]["use_judge_id"]
    
    print('Starting prediction')

    for i, data in enumerate(evalloader, 0):
        *inputs, labels, filenames = data
        wav, wav_length, f0_start, f0_variation, f0_origin = inputs
        wav = wav.to(device)
        wav_length = wav_length.to(device)
        f0_start = f0_start.to(device)
        f0_variation = f0_variation.to(device)
        f0_origin = f0_origin.to(device)
        labels = labels.to(device)
        judge_id = None
        if use_judge_id is True:
            judge_id = torch.zeros_like(wav_length).to(device)
        batch = {}
        batch.update(
            audio=wav,
            audio_length=wav_length,
            f0_start=f0_start,
            f0_variation=f0_variation,
            f0_origin=f0_origin,
            judge_id = judge_id
        )
        outputs = model(**batch)
        frame_loss = 0
        utt_loss = utt_loss_func(outputs["utt_score"], labels, frame_level=False)
        if model_config["use_frame_loss"] is True:
            frame_loss = frame_loss_func(outputs["frame_score"], labels, frame_level=True)
        loss = utt_loss + frame_loss * model_config["alpha_frame"]

        total_loss += loss.item()
        
        output = outputs["utt_score"].cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1

    metrics_answer = {}
    metrics_answer["utterance"] = {}
    metrics_answer["system"] = {}

    true_MOS = {}
    evalf = open(evallist, 'r')
    for line in evalf:
        parts = line.strip().split(',')
        uttID = parts[0]
        MOS = float(parts[1])
        true_MOS[uttID] = MOS

    ## compute correls.
    sorted_uttIDs = sorted(predictions.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
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
    # print('[UTTERANCE] Test error= %f' % MSE)
    # print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
    # print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])
    # print('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    ### SYSTEM
    true_sys_MOS_avg = {}
    csv_file = open(system_csv_path, 'r')
    csv_file.readline()  ## skip header
    for line in csv_file:
        parts = line.strip().split(',')
        sysID = parts[0]
        MOS = float(parts[1])
        true_sys_MOS_avg[sysID] = MOS

    # utt_mos_file = open(f"/data3/tyx/z_/mos/val_mos_utt.txt", "w", encoding="utf-8")
    pred_sys_MOSes = {}
    for uttID in sorted_uttIDs:
        sysID = systemID(uttID)
        noop = pred_sys_MOSes.setdefault(sysID, [])
        pred_sys_MOSes[sysID].append(predictions[uttID])
        # utt_mos_file.write(f'{uttID},{predictions[uttID]}\n')

    pred_sys_MOS_avg = {}
    for k, v in pred_sys_MOSes.items():
        avg_MOS = sum(v) / (len(v) * 1.0)
        pred_sys_MOS_avg[k] = avg_MOS

    ## make lists sorted by system
    pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
    sys_p = []
    sys_t = []
    for sysID in pred_sysIDs:
        sys_p.append(pred_sys_MOS_avg[sysID])
        sys_t.append(true_sys_MOS_avg[sysID])

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
    # print('[SYSTEM] Test error= %f' % MSE)
    # print('[SYSTEM] Linear correlation coefficient= %f' % LCC[0][1])
    # print('[SYSTEM] Spearman rank correlation coefficient= %f' % SRCC[0])
    # print('[SYSTEM] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    make_dir(os.path.join(answer_dir, "metrics"))
    with open(os.path.join(answer_dir, "metrics", fn + ".json"), "w", encoding="utf-8") as f:
        json.dump(metrics_answer, f, cls=NumpyEncoder, indent=4)
        print('Metrics writen in {}'.format(os.path.join(answer_dir, "metrics", fn + ".json")))

    make_dir(os.path.join(answer_dir, "utt_mos"))
    with open(os.path.join(answer_dir, "utt_mos", fn + ".txt"), "w", encoding="utf-8") as f:
        for k, v in predictions.items():
            outl = k.split('.')[0] + ',' + str(v) + '\n'
            f.write(outl)
        f.close()
        print('Utt MOS writen in {}'.format(os.path.join(answer_dir, "utt_mos", fn + ".txt")))

if __name__ == '__main__':
    main()
