import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer


from mamba_ssm import Mamba
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    # a = self.get_parameter_number()
    #
    # def get_parameter_number(self):
    #     """
    #     Number of model parameters (without stable diffusion)
    #     """
    #     total_num = sum(p.numel() for p in self.parameters())
    #     trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     trainable_ratio = trainable_num / total_num
    #
    #     print('total_num:', total_num)
    #     print('trainable_num:', total_num)
    #     print('trainable_ratio:', trainable_ratio)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
if __name__ == "__main__":
    class Config:
        seq_len = 96
        pred_len = 24
        d_model = 64
        d_state = 16
        e_layers = 2
        d_ff = 128
        dropout = 0.1
        activation = 'gelu'
        embed = 'timeF'
        freq = 'h'
        use_norm = True
        output_attention = False
        class_strategy = None  # not used in this script

    configs = Config()
    
    B = 4   # batch size
    N = 8   # number of variables (features)
    T = configs.seq_len
    S = configs.pred_len

    # dummy input
    x_enc = torch.randn(B, T, N).cuda()
    x_mark_enc = torch.randn(B, T, 4).cuda()  # e.g., time features like hour, day, etc.
    x_dec = torch.randn(B, S, N)  .cuda()     # not used in forecast, but kept for interface
    x_mark_dec = torch.randn(B, S, 4).cuda()

    model = Model(configs).cuda()
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    print(f"Output shape: {out.shape}")  # Expected: [B, pred_len, N]
