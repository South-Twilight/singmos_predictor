import os

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from singmos.utils import calc_f0_variation

class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list, use_judge_id=False):
        self.mos_lookup = {}
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            if use_judge_id:
                wavname = wavname + "_" + parts[2]
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())
        self.use_judge_id = use_judge_id

        
    def __getitem__(self, idx):
        """
            wav: [1, L]
            f0_start: [1]
            f0_variation: [1, T]
            f0_origin: [1, T]
        """
        wavname = self.wavnames[idx]
        wav_path = wavname
        if self.use_judge_id:
            items = wavname.split('_')
            speaker_id = int(items[-1])
            wav_path = '_'.join(items[:-1])
        wavpath = os.path.join(self.wavdir, wav_path)
        wav = torchaudio.load(wavpath)[0]
        f0_start, f0_variation, f0_origin = calc_f0_variation(
            wav[0],
            sampling_rate=16000,
            use_log_f0=False,
            use_continuous_f0=False,
            use_discrete_f0=True,
        )
        score = self.mos_lookup[wavname]
        if self.use_judge_id:
            return wav, f0_start, f0_variation, f0_origin, score, wavname, speaker_id
        else:
            return wav, f0_start, f0_variation, f0_origin, score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        if not self.use_judge_id:
            wavs, f0_starts, f0_variations, f0_origins, scores, wavnames = zip(*batch)
        else:
            wavs, f0_starts, f0_variations, f0_origins, scores, wavnames, speaker_ids = zip(*batch)

        wavs = list(wavs)
        # padding wavs
        wav_max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        
        output_wavs = []
        wav_length = []
        for wav in wavs:
            wav_length.append(wav.shape[1])
            amount_to_pad = wav_max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        wav_length = torch.tensor(wav_length, dtype=torch.long)
        
        f0_var_max_len = max(f0_variations, key = lambda x : x.shape[0]).shape[0]
        output_f0_variations = []
        for f0_var in f0_variations:
            amount_to_pad = f0_var_max_len - f0_var.shape[0]
            padded_f0_var = torch.nn.functional.pad(f0_var, (0, amount_to_pad), 'constant', 0)
            output_f0_variations.append(padded_f0_var)
        output_f0_variations = torch.stack(output_f0_variations, dim=0)
        
        f0_max_len = max(f0_origins, key = lambda x : x.shape[0]).shape[0]
        output_f0 = []
        for f0 in f0_origins:
            amount_to_pad = f0_max_len - f0.shape[0]
            padded_f0 = torch.nn.functional.pad(f0, (0, amount_to_pad), 'constant', 0)
            output_f0.append(padded_f0)
        output_f0 = torch.stack(output_f0, dim=0)
        
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        f0_starts  = torch.tensor(f0_starts)

        if self.use_judge_id:
            speaker_ids = torch.tensor(speaker_ids)
            wav_names = []
            for wavname in wav_names:
                items = wavname.split("_")
                wav_names.append("_".join(items[-1]))
            return output_wavs, wav_length, f0_starts, output_f0_variations, output_f0, scores, wav_names, speaker_ids
        else:
            return output_wavs, wav_length, f0_starts, output_f0_variations, output_f0, scores, wavnames

