import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class PatchTST_agent(nn.Module):

    def __init__(self, configs):
        super(PatchTST_agent, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.d_model = configs.d_model
        self.patch_len = 16
        self.patch_num = self.seq_len//self.patch_len
        self.basis_num = min(self.patch_num,6)
        self.time_base = nn.Linear(self.patch_num, self.basis_num)
        self.enc_embedding = nn.Linear(self.patch_len, configs.d_model)
        # Encoder
              # Encoder Layers as ModuleList
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model,
                    configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ])
        
        self.norm = nn.LayerNorm(configs.d_model)  # optional
        
        self.projection = nn.Linear(configs.d_model*self.basis_num, self.pred_len, bias=True)

        self.enc_in = configs.enc_in
    def forward(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - means)/stdev

        B, T, C = x_enc.shape
        
        x_enc = x_enc.view(B,self.patch_num,self.patch_len,C)

        x_enc = x_enc.permute(0,3,2,1) #  B C P N
        x_enc = self.time_base(x_enc).permute(0,1,3,2)
        enc_out = self.enc_embedding(x_enc)
        enc_out = enc_out.view(B*C,self.basis_num,self.d_model)
    
        for layer in self.encoder_layers:
            enc_out, _ = layer(enc_out)

        enc_out = self.norm(enc_out)
        enc_out = enc_out.view(B*C,self.basis_num*self.d_model)
        dec_out = self.projection(enc_out).view(B,self.enc_in,self.pred_len).permute(0, 2, 1)#[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out

    def initial_embed(self, x_enc):
        #print(x_enc.shape)
        # Normalization from Non-stationary Transformer
        self.means = x_enc.mean(1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - self.means)/self.stdev

        # Embedding
        B, T, C = x_enc.shape
        #print(x_enc.shape)
        x_enc = x_enc.view(B,self.patch_num,self.patch_len,C)

        x_enc = x_enc.permute(0,3,2,1) #  B C P N
        x_enc = self.time_base(x_enc).permute(0,1,3,2)
        enc_out = self.enc_embedding(x_enc)
        enc_out = enc_out.view(B*C,self.basis_num,self.d_model)
        return enc_out

    def project(self, enc_out):
        BC, basis_num, d_model = enc_out.shape
        #print(enc_out.shape)
        enc_out = enc_out.view(BC,self.basis_num*self.d_model)
        dec_out = self.projection(enc_out).view(-1,self.enc_in,self.pred_len).permute(0, 2, 1)#[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * self.stdev
        dec_out = dec_out + self.means
        return dec_out


    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_models = 4
        

        self.models = nn.ModuleList([PatchTST_agent(configs) for _ in range(self.num_models)])
        
        self.aggregation = nn.Linear(self.num_models, 1)

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark):
        """
        x: [B, T, C]
        return: [B, T_pred, C]
        """
        # 存储4个模型的输出：每个是 [B, T_pred, C]
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(out)

        # 拼接输出: [B, T_pred, C, 4]
        stacked = torch.stack(outputs, dim=-1)
        
        # 聚合: Linear(4 -> 1), 输出 shape: [B, T_pred, C]
        out = self.aggregation(stacked).squeeze(-1)

        return out
