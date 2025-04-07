import os
import argparse
import logging

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

from s3prl.nn import S3PRLUpstream

from .utils import make_non_pad_mask

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


class Projection_Layer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        activation_func,
        output_type="scalar",
        mos_clip=True,
    ):
        super(Projection_Layer, self).__init__()

        self.output_type = output_type
        self.mos_clip = mos_clip
        if output_type == "scalar":
            output_dim = 1
            if mos_clip:
                self.proj = nn.Tanh()
        elif output_type == "categorical":
            output_dim = 5
        else:
            raise NotImplementedError("wrong output_type: {}".format(output_type))
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation_func(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        output = self.net(x)
        # mos range clip
        if self.output_type == "scalar" and self.mos_clip:
            return self.proj(output) * 2.0 + 3
        else:
            return output


class MOS_Predictor(nn.Module):
    def __init__(
        self,
        ssl_model_type,
        hdim = 128,
        # f0 embedding
        use_f0 = False,
        use_f0_var = False,
        f0_range = 801,
        # judge embedding
        use_judge_id = False,
        use_lstm = False,
        judge_num = 50,
        # projection layer
        activate_func = "relu",
        output_type = "scalar",
        mos_clip = True,
        # loss related:
        loss_type = "L2",
        use_margin = True,
        margin = 0.1,
        use_frame_level = False,
        alpha_judge = 0.8,
        alpha_frame = 0.5,
    ) -> None:
        """ MOS Predictor for Singing:
            pitch_num (int): Max range of pitch
        """
        super(MOS_Predictor, self).__init__()

        self.ssl_model, feature_dim = load_ssl_model_s3prl(ssl_model_type)

        # F0 Embedding
        self.use_f0 = use_f0
        if use_f0 or use_f0_var:
            self.f0_embedding = nn.Embedding(num_embeddings=f0_range, embedding_dim=hdim)
            if self.use_f0:
                self.f0_blstm = nn.LSTM(input_size=hdim, hidden_size=hdim, num_layers=1, bidirectional=True, batch_first=True)
                feature_dim += hdim * 2 # bidirectional=True

        # F0 Variation Embedding
        self.use_f0_var = use_f0_var
        if use_f0_var is True:
            # range of f0_var is transfered from [-B, B] to [0, 2 * B]
            self.f0_var_embedding = nn.Embedding(num_embeddings=f0_range * 2, embedding_dim=hdim)
            self.f0_var_blstm = nn.LSTM(input_size=hdim, hidden_size=hdim, num_layers=1, bidirectional=True, batch_first=True)
            feature_dim += hdim * 2 # bidirectional=True

        # Judge Embedding
        self.use_judge_id = use_judge_id 
        if use_judge_id is True:
            self.judge_embedding = nn.Embedding(num_embeddings=judge_num, embedding_dim=hdim)

        # Output layer
        if activate_func == "relu":
            acti_func = nn.ReLU
        else:
            raise NotImplementedError("wrong activate_func: {}".format(activate_func))

        # Mean score projection layer
        self.mean_proj = Projection_Layer(
            in_dim=feature_dim,
            hidden_dim=64,
            activation_func=acti_func,
            output_type=output_type,
            mos_clip=mos_clip,
        )
        # Bias score projection layer
        if use_judge_id is True:
            self.use_lstm = use_lstm
            if use_lstm:
                self.bias_blstm = nn.LSTM(input_size=feature_dim + hdim, hidden_size=hdim, num_layers=1, bidirectional=True, batch_first=True)
                bias_input_dim = hdim * 2
            else:
                bias_input_dim = feature_dim
            self.bias_proj = Projection_Layer(
                in_dim=bias_input_dim,
                hidden_dim=64,
                activation_func=acti_func,
                output_type=output_type,
                mos_clip=mos_clip,
            )
        
        self.use_frame_level = use_frame_level
        self.alpha_frame = alpha_frame
        self.alpha_judge = alpha_judge
        self.mean_loss = MOS_Loss(loss_type=loss_type, use_margin=use_margin, margin=margin)
        if use_judge_id is True:
            self.bias_loss = MOS_Loss(loss_type=loss_type, use_margin=use_margin, margin=margin)

    
    def forward(
        self,
        audio,
        audio_length,
        f0_variation = None,
        f0 = None,
        judge_id = None,
        # for loss
        mean_score = None,
        judge_score = None,
    ):
        """
        Forward function

        Args:
            audio (torch.Tensor): Wav feature, shape [B, 1, T]
            audio_length (torch.Tensor): Wav length, shape [B]
            f0_start (torch.Tensor, optional): Start element of f0 sequence, assists f0 variance sequence, shape [B]
            f0_variation (torch.Tensor, optional): Variation of f0 sequence, shape [B, T]
            f0 (torch.Tensor, optional): f0 sequence, shape [B, T]
            judge_id (torch.Tensor, optional): Judge ID for MOS, shape [B]

        Returns:
            dict: Dictionary containing frame and utterance level mean scores, and optionally bias scores if judge_id is provided.
        """
        ssl_feature = self.ssl_model(audio, audio_length)
        # ssl_feature = self.ssl_model(audio)
        T_len = ssl_feature.shape[1]
        
        if self.use_f0 is True:
            f0_feature = self.f0_embedding(f0)
            f0_feature, _ = self.f0_blstm(f0_feature)
            assert T_len == f0_feature.shape[1]
            
        if self.use_f0_var:
            f0_var_feature = self.f0_var_embedding(f0_variation)
            f0_var_feature, _ = self.f0_var_blstm(f0_var_feature)
            assert T_len == f0_var_feature.shape[1]
            
        x = ssl_feature
        if self.use_f0 is True:
            x = torch.cat((x, f0_feature), dim=2)
        if self.use_f0_var:
            x = torch.cat((x, f0_var_feature), dim=2)
        
        masks = make_non_pad_mask(audio_length)
        # mean net 
        mean_input = x 
        mean_frame_score = self.mean_proj(mean_input)
        mean_utt_score = torch.mean(mean_frame_score, dim=1)
        mean_loss = self.mean_loss(mean_utt_score, mean_score, frame_level=False)
        if self.use_frame_level:
            mean_frame_loss = self.mean_loss(mean_frame_score, mean_score, masks, frame_level=True)
        # judge net
        if self.use_judge_id is True:
            judge_feature = self.judge_embedding(judge_id)
            bias_input = torch.cat((mean_input, judge_feature.unsqueeze(1).repeat(1, T_len, 1)), dim=2)
            if self.use_lstm:
                bias_input, (_, _) = self.bias_blstm(bias_input)
            bias_frame_score = self.bias_proj(bias_input)
            bias_utt_score = torch.mean(bias_frame_score, dim=1)
            bias_loss = self.bias_loss(bias_utt_score, judge_score, frame_level=False)
            if self.use_frame_level:
                bias_frame_loss = self.mean_loss(bias_frame_score, judge_score, masks, frame_level=True)
        
        # loss calculate
        ret_val = {
            "mean_utt_score": mean_utt_score,
        }
        stats = {}
        loss = 0
        stats["mean_loss"] = mean_loss
        loss += mean_loss
        if self.use_frame_level:
            stats["mean_frame_loss"] = mean_frame_loss
            loss += mean_frame_loss * self.alpha_frame
            ret_val.update(mean_frame_score=mean_frame_score)
        if self.use_judge_id is True:
            judge_loss = 0
            stats["bias_loss"] = bias_loss
            judge_loss += bias_loss
            ret_val.update(bias_utt_score=bias_utt_score)
            if self.use_frame_level:
                stats["bias_frame_loss"] = bias_frame_loss
                judge_loss += bias_frame_loss * self.alpha_frame
                ret_val.update(bias_frame_score=bias_frame_score)
            loss += judge_loss * self.alpha_judge
        stats["loss"] = loss
        return loss, stats, ret_val


class MOS_Loss(nn.Module):
    def __init__(
        self,
        loss_type="L2",
        use_margin=True,
        margin=0.1,
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

    def forward(
        self,
        pred_score,
        gt_score,
        masks=None,
        frame_level=False
    ):
        """
        Args:
            pred_score: frame score [B, T] / utterance score [B]
            gt_score: [B]
            frame_level: whether is frame level
        """
        assert len(pred_score.shape) <= 2
        if frame_level is True:
            gt_score = gt_score.unsqueeze(1).expand(pred_score.shape)
            if masks is not None:
                pred_score = pred_score.masked_select(masks)
                gt_score = gt_score.masked_select(masks)

        if self.use_margin is True:
            diff = torch.abs(pred_score - gt_score)
            zero_mask = torch.zeros(pred_score.shape, dtype=torch.bool).to(diff.device)
            zero_mask = torch.where(diff < self.margin, torch.tensor(True, dtype=torch.bool, device=diff.device), zero_mask)
            pred_score = pred_score.masked_fill(zero_mask, 0.0)
            gt_score = gt_score.masked_fill(zero_mask, 0.0)

        Loss = self.loss(pred_score, gt_score)

        return Loss
