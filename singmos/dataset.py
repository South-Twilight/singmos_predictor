import os

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data.dataset import Dataset

from singmos2.utils import calc_f0_variation
from singmos2.utils import pad_sequence

class MOSDataset(Dataset):
    def __init__(
        self, 
        utt_list, 
        score_infos, 
        datadir,
        use_judge_id=False,
        use_f0=False,
    ):
        self.use_judge_id = use_judge_id
        self.use_f0 = use_f0
        score_info = {idx: score_infos[idx] for idx in utt_list if idx in score_infos}
        sysnames = {}
        uttnames = []
        wavnames = [] # judge wavname
        wavs = {}
        scores = {}
        mean_scores = {}
        for idx in score_info.keys():
            mean_id = idx + "_0"
            wavnames.append(mean_id)
            scores[mean_id] = float(score_info[idx]["score"]["mos"])
            mean_scores[mean_id] = float(score_info[idx]["score"]["mos"])
            wavs[mean_id] = os.path.join(datadir, score_info[idx]["wav"])
            sysnames[score_info[idx]["sys_id"]] = True
            uttnames.append(idx)
            if use_judge_id:
                for judge, judge_score in zip(score_info[idx]["score"]["judges"], score_info[idx]["score"]["scores"]):
                    judge_id = idx + f"_{judge}"
                    wavnames.append(judge_id)
                    scores[judge_id] = float(judge_score)
                    mean_scores[judge_id] = float(score_info[idx]["score"]["mos"])
                    wavs[judge_id] = os.path.join(datadir, score_info[idx]["wav"])
        self.sysnames = sorted(sysnames.keys())
        self.uttnames = sorted(uttnames)
        self.wavnames = sorted(wavnames)
        self.wavs = {k: v for k, v in sorted(wavs.items(), key=lambda x: x[0])}
        self.scores = {k: v for k, v in sorted(scores.items(), key=lambda x: x[0])}
        self.mean_scores = {k: v for k, v in sorted(mean_scores.items(), key=lambda x: x[0])}

        
    def __getitem__(self, idx):
        """
        return value: 
            wav: wav feature, [1, L]
            f0_start: [1]
            f0_variation: [1, T]
            f0_origin: [1, T]
        """
        wavname = self.wavnames[idx]
        wav = torchaudio.load(self.wavs[wavname])[0]
        score = torch.tensor([self.scores[wavname]], dtype=torch.float32)
        mean_score = torch.tensor([self.mean_scores[wavname]], dtype=torch.float32)

        ret_dict = {}
        items = wavname.split('_')
        wavname = "".join(items[:-1])
        judge_id = int(items[-1])
        ret_dict.update(judge_id=judge_id)
        if self.use_f0:
            f0_start, f0_variation, f0 = calc_f0_variation(
                wav[0],
                sampling_rate=16000,
                use_log_f0=False,
                use_continuous_f0=False,
                use_discrete_f0=True,
            )
            ret_dict.update(f0_start=f0_start)
            ret_dict.update(f0_variation=f0_variation)
            ret_dict.update(f0=f0)
        return wavname, wav, score, mean_score, ret_dict
    

    def __len__(self):
        return len(self.wavnames)

    def get_sys_names(self):
        return self.sysnames
    
    def get_utt_names(self):
        return self.uttnames

    def collate_fn(self, batch):  ## zero padding
        wavnames, wavs, scores, mean_scores, ret_dicts = zip(*batch)

        # padding wavs
        collate_wavs = pad_sequence(wavs)
        wavs_length = torch.tensor([seq.shape[-1] for seq in wavs])
        T_max = torch.max(wavs_length, dim=0)
        collate_scores = pad_sequence(scores)
        collate_mean_scores = pad_sequence(mean_scores)
        collate_ret_dict = {}

        if "judge_id" in ret_dicts[0]:
            judge_ids = [rd['judge_id'] for rd in ret_dicts]
            collate_ret_dict["judge_id"] = torch.tensor(judge_ids)

        if "f0_start" in ret_dicts[0]:
            f0_start = [rd['f0_start'] for rd in ret_dicts]
            collate_ret_dict["f0_start"] = torch.tensor(f0_start)
        
        if "f0_variation" in ret_dicts[0]:
            f0_variation = [rd['f0_variation'] for rd in ret_dicts]
            collate_ret_dict["f0_variation"] = pad_sequence(f0_variation, max_length=T_max)

        if "f0" in ret_dicts[0]:
            f0= [rd['f0_'] for rd in ret_dicts]
            collate_ret_dict["f0"] = pad_sequence(f0, max_length=T_max)
        
        return wavnames, collate_wavs, wavs_length, collate_scores, collate_mean_scores, collate_ret_dict
