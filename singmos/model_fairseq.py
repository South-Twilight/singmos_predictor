import os
import argparse
import fairseq

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils import calc_f0_variation

import logging

class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            mos = float(parts[1])
            self.mos_lookup[wavname] = mos

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

        
    def __getitem__(self, idx):
        """
            wav: [1, L]
            f0_start: [1]
            f0_variation: [1, T]
            f0_origin: [1, T]
        """
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        f0_start, f0_variation, f0_origin = calc_f0_variation(
            wav[0],
            sampling_rate=16000,
            use_log_f0=False,
            use_continuous_f0=False,
            use_discrete_f0=True,
        )
        score = self.mos_lookup[wavname]
        return wav, f0_start, f0_variation, f0_origin, score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        wavs, f0_starts, f0_variations, f0_origins, scores, wavnames = zip(*batch)
        wavs = list(wavs)
        # padding wavs
        wav_max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        
        output_wavs = []
        for wav in wavs:
            amount_to_pad = wav_max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        
        f0_var_max_len = max(f0_variations, key = lambda x : x.shape[0]).shape[0]
        output_f0_variations = []
        for f0_var in f0_variations:
            amount_to_pad = f0_var_max_len - f0_var.shape[0]
            padded_f0_var = torch.nn.functional.pad(f0_var, (0, amount_to_pad), 'constant', 0)
            output_f0_variations.append(padded_f0_var)
        output_f0_variations = torch.stack(output_f0_variations, dim=0)
        
        f0_max_len = max(f0_variations, key = lambda x : x.shape[0]).shape[0]
        output_f0 = []
        for f0 in f0_origins:
            amount_to_pad = f0_max_len - f0.shape[0]
            padded_f0 = torch.nn.functional.pad(f0, (0, amount_to_pad), 'constant', 0)
            output_f0.append(padded_f0)
        output_f0 = torch.stack(output_f0, dim=0)
        
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        f0_starts  = torch.tensor(f0_starts)
        return output_wavs, f0_starts, output_f0_variations, output_f0, scores, wavnames

ssl_model_list = [
    "hubert_base",
    "hubert_large",
    "wav2vec2_small",
    "wav2vec2_large",
    "xlsr_base",
]

class SSL_Model(nn.Module):
    def __init__(
        self, 
        model_path, 
        ssl_out_dim,
        ssl_model_name = "wav2vec2_small",
    ):
        super(SSL_Model, self).__init__()
        
        if ssl_model_name in ssl_model_list:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
            ssl_model = model[0]
            ssl_model.remove_pretraining_modules()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        
    def forward(self, wav):
        wav = wav.squeeze(1)  # [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        return x


class Sing_MOS_Predictor(nn.Module):
    def __init__(
        self,
        model_path,
        ssl_dim,
        ssl_model_name = "wav2vec2_small",
        use_f0 = False,
        use_f0_variation = False,
        use_judge_id = False,
        judge_num = 20,
        use_speaker_id = False,
        speaker_num = 20,
        pitch_num = 801,
        hdim = 128,
        use_lstm = False,
    ):
        """ MOS Predictor for Singing:
            pitch_num (int): Max range of pitch
                    
        """
        super(Sing_MOS_Predictor, self).__init__()
        
        feature_dim = ssl_dim
        self.ssl_model = SSL_Model(
            model_path,
            ssl_dim,
            ssl_model_name,
        )
        # F0 
        self.use_f0 = use_f0
        if use_f0 is True:
            self.f0_embedding = torch.nn.Embedding(
                num_embeddings=pitch_num,
                embedding_dim=hdim,
            )
            feature_dim += hdim
        # F0 Variation
        self.use_f0_variation = use_f0_variation
        if use_f0_variation is True:
            self.f0_start_embedding = torch.nn.Embedding(
                num_embeddings=pitch_num,
                embedding_dim=hdim,
            )
            self.f0_variation_embedding = torch.nn.Embedding(
                num_embeddings=pitch_num * 2, # [-B, B] -> [0, 2 * B]
                embedding_dim=hdim,
            )
            feature_dim += hdim * 2
        # Judge
        self.use_judge_id = use_judge_id 
        if use_judge_id is True:
            self.judge_embedding = torch.nn.Embedding(
                num_embeddings=judge_num,
                embedding_dim=hdim,
            )
            feature_dim += hdim
        # Speaker
        self.use_speaker_id = use_speaker_id 
        if use_speaker_id is True:
            self.speaker_embedding = torch.nn.Embedding(
                num_embeddings=speaker_num,
                embedding_dim=hdim,
            )
            feature_dim += hdim
        
        self.use_lstm = use_lstm
        if self.use_lstm is True:    
            self.blstm = torch.nn.LSTM(
                input_size = feature_dim, 
                hidden_size = hdim, 
                num_layers = 1, 
                bidirectional=True, 
                batch_first=True
            )
            self.linear = torch.nn.Linear(
                hdim * 2, 1
            )
        else:
            self.linear = torch.nn.Linear(
                feature_dim, 1
            )
        
    
    def forward(
        self,
        audio,
        f0_start = None,
        f0_variation = None,
        f0_origin = None,
        judge_id = None,
        speaker_id = None,
    ):
        ssl_feature = self.ssl_model(audio)
        
        T_len = ssl_feature.shape[1]
        
        if self.use_f0 is True:
            f0_feature = self.f0_embedding(f0_origin)
            T_len = min(T_len, f0_origin.shape[1])
            
        if self.use_f0_variation:
            f0_start_feature = self.f0_start_embedding(f0_start)
            f0_variation_feature = self.f0_variation_embedding(f0_variation)
            T_len = min(T_len, f0_variation_feature.shape[1])
            
        x = ssl_feature[:, :T_len, :]
        if self.use_f0 is True:
            f0_feature = f0_feature[:, : T_len, :]
            x = torch.cat((x, f0_feature), dim=2)
        if self.use_f0_variation:
            f0_variation_feature = f0_variation_feature[:, :T_len, :]
            f0_start_feature = f0_start_feature.unsqueeze(1).repeat(1, T_len, 1) # Same ize in dim=0,1
            x = torch.cat((x, f0_start_feature), dim=2)
            x = torch.cat((x, f0_variation_feature), dim=2)
        
        if judge_id is not None and self.use_judge_id is True:
            speaker_feature = self.speaker_embedding(speaker_id)
            x = torch.cat((x, speaker_feature.unsqueeze(1).repeat(1, T_len, 1)), dim=2)
        if speaker_id is not None and self.use_speaker_id is True:
            judge_feature = self.judge_embedding(judge_id)
            x = torch.cat((x, judge_feature.unsqueeze(1).repeat(1, T_len, 1)), dim=2)
        
        if self.use_lstm:
            lstm_out, _ = self.blstm(x)
            pred_score = self.linear(lstm_out[:, -1, :])
            pred_score = pred_score.squeeze(1)
        else:
            pred_score = self.linear(x)
            pred_score = torch.mean(pred_score, dim=1)
            pred_score = pred_score.squeeze(1)
        return pred_score
