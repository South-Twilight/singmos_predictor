import os
import argparse
import logging

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

from s3prl.nn import S3PRLUpstream

import random
random.seed(1984)

ssl_model_list = [
    "wavlm_base",
    "wavlm_large",
    "wav2vec2_base_960",
    "wav2vec2_large_lv60_cv_swbd_fsh",
    "hubert_base",
    "hubert_large_ll60k",
    "xls_r_300m",
]

def load_ssl_model_s3prl(ssl_model_type, use_proxy = True):
    assert ssl_model_type in ssl_model_list, f"***ERROR***: {ssl_model_type} is not support, please check ssl_model_list."
    if "base" in ssl_model_type:
        SSL_OUT_DIM = 768
    elif "large" in ssl_model_type or ssl_model_type in ["xls_r_300m"]:
        SSL_OUT_DIM = 1024
    if use_proxy:
        os.environ['http_proxy'] = 'http://127.0.0.1:7890'
        os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    ssl_model = S3PRLUpstream(ssl_model_type)
    return SSL_Model(ssl_model, SSL_OUT_DIM), SSL_OUT_DIM


class SSL_Model(nn.Module):
    def __init__(
        self, 
        ssl_model, 
        ssl_out_dim,
    ):
        super(SSL_Model, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_out_dim = ssl_out_dim
        
    def forward(self, wav, wav_length):
        wav = wav.squeeze(1)  # [B, T]
        ssl_features, ssl_lens = self.ssl_model(wav, wav_length)
        return ssl_features[-1]


class MOS_Predictor(nn.Module):
    def __init__(
        self,
        ssl_model_type,
        # additional info
        use_f0 = False,
        use_f0_var = False,
        use_judge_id = False,
        use_speaker_id = False,
        # parameters
        hdim = 128,
        judge_num = 20,
        speaker_num = 20,
        pitch_num = 801,
        use_lstm = False,
    ) -> None:
        """ MOS Predictor for Singing:
            pitch_num (int): Max range of pitch
        """
        super(MOS_Predictor, self).__init__()

        self.ssl_model, feature_dim = load_ssl_model_s3prl(ssl_model_type)

        # F0 Embedding
        self.use_f0 = use_f0
        if use_f0 or use_f0_var:
            self.f0_embedding = torch.nn.Embedding(
                num_embeddings=pitch_num,
                embedding_dim=hdim,
            )
            if self.use_f0:
                self.f0_blstm = torch.nn.LSTM(
                    input_size = hdim, 
                    hidden_size = hdim, 
                    num_layers = 1, 
                    bidirectional=True, 
                    batch_first=True 
                )
                feature_dim += hdim * 2 # bidirectional=True

        # F0 Variation Embedding
        self.use_f0_var = use_f0_var
        if use_f0_var is True:
            self.f0_var_embedding = torch.nn.Embedding(
                num_embeddings=pitch_num * 2, # [-B, B] -> [0, 2 * B]
                embedding_dim=hdim,
            )
            self.f0_var_blstm = torch.nn.LSTM(
                input_size = hdim, 
                hidden_size = hdim, 
                num_layers = 1, 
                bidirectional=True, 
                batch_first=True 
            )
            feature_dim += hdim * 2 # bidirectional=True

        # Judge Embedding
        self.use_judge_id = use_judge_id 
        if use_judge_id is True:
            self.judge_embedding = torch.nn.Embedding(
                num_embeddings=judge_num,
                embedding_dim=hdim,
            )
            feature_dim += hdim

        # Speaker Embedding
        self.use_speaker_id = use_speaker_id 
        if use_speaker_id is True:
            self.speaker_embedding = torch.nn.Embedding(
                num_embeddings=speaker_num,
                embedding_dim=hdim,
            )
            feature_dim += hdim
        
        # Output layer
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
        audio_length,
        f0_start = None,
        f0_variation = None,
        f0_origin = None,
        judge_id = None,
        speaker_id = None,
    ):
        ssl_feature = self.ssl_model(audio, audio_length)
        # ssl_feature = self.ssl_model(audio)
        T_len = ssl_feature.shape[1]
        
        if self.use_f0 is True:
            f0_feature = self.f0_embedding(f0_origin)
            f0_feature, _ = self.f0_blstm(f0_feature)
            T_len = min(T_len, f0_feature.shape[1])
            
        if self.use_f0_var:
            f0_start_feature = self.f0_embedding(f0_start)
            f0_var_feature = self.f0_var_embedding(f0_variation)
            f0_var_feature = torch.cat((f0_start_feature.unsqueeze(1), f0_var_feature), dim=1)
            f0_var_feature, _ = self.f0_var_blstm(f0_var_feature)
            T_len = min(T_len, f0_var_feature.shape[1])
            
        x = ssl_feature[:, :T_len, :]
        if self.use_f0 is True:
            f0_feature = f0_feature[:, : T_len, :]
            x = torch.cat((x, f0_feature), dim=2)
        if self.use_f0_var:
            f0_var_feature = f0_var_feature[:, :T_len, :]
            x = torch.cat((x, f0_var_feature), dim=2)
        
        # if speaker_id is not None and self.use_speaker_id is True:
        #     speaker_feature = self.speaker_embedding(speaker_id)
        #     x = torch.cat((x, speaker_feature.unsqueeze(1).repeat(1, T_len, 1)), dim=2)

        if judge_id is not None and self.use_judge_id is True:
            judge_feature = self.judge_embedding(judge_id)
            x = torch.cat((x, judge_feature.unsqueeze(1).repeat(1, T_len, 1)), dim=2)
        
        if self.use_lstm:
            lstm_out, _ = self.blstm(x)
            frame_score = self.linear(lstm_out).squeeze(-1)
            utt_score = frame_score[:, -1]
        else:
            frame_score = self.linear(x).squeeze(-1)
            utt_score = torch.mean(frame_score, dim=1)
        return {
            "utt_score": utt_score, 
            "frame_score": frame_score,
        }


class MOS_Loss(nn.Module):
    def __init__(
        self, loss_type="L2", use_margin=True, margin=0.1
    ):
        super(MOS_Loss, self).__init__()

        if loss_type == "L2":
            self.loss = torch.nn.MSELoss(reduction='mean')
        elif loss_type == "L1":
            self.loss = torch.nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"***ERROR*** {loss_type} is not support.")
        self.use_margin = use_margin
        self.margin = margin 


    def forward(self, pred_score, gt_score, frame_level=False):
        """
        pred_score: frame score [B, T] / utterance score [B]
        gt_score: [B]
        frame_level: whether is frame level
        """
        assert len(pred_score.shape) <= 2
        # pred_score: [B, 1]/[B, T], gt_score: [B]
        if frame_level is True:
            gt_score = gt_score.unsqueeze(1).expand(pred_score.shape)

        if self.use_margin is True:
            diff = torch.abs(pred_score - gt_score)
            zero_mask = torch.zeros(pred_score.shape, dtype=torch.bool).to(diff.device)
            zero_mask = torch.where(diff < self.margin, torch.tensor(True, dtype=torch.bool, device=diff.device), zero_mask)
            pred_score = pred_score.masked_fill(zero_mask, 0.0)
            gt_score = gt_score.masked_fill(zero_mask, 0.0)

        Loss = self.loss(pred_score, gt_score)

        return Loss
