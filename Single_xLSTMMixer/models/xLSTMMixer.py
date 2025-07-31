import enum
from typing import Literal
from torch import Tensor, nn
import torch


from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig

from einops.layers.torch import Rearrange

from einops import rearrange, repeat, pack, unpack
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    mLSTMBlockConfig,
)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class AblationMode(enum.Enum):
    FULL = "full"
    NLINEAR_ONLY = "nlinear-only"
    SLSTM_ONLY = "sLSTM-only"
    SLSTM_BACKCASTING = "sLSTM+Back"
    SLSTM_MEMORY = "sLSTM+Mem"
    SLSTM_MEMORY_BACK = "sLSTM+Mem+Back"
    FULL_MEM1 = "full-mem1"
    FULL_MEM2 = "full-mem2"
    FULL_MEM3 = "full-mem3"
    FULL_MEM4 = "full-mem4"
    FULL_NO_MEMORY = "full-no-memory"
    FULL_NO_BACKCAST = "full-no-backcast"
    FULL_TIME = "full-time"
    FULL_NO_MEMORY_BACKCAST = "full-no-memory-backcast"
    FULL_TIME_NO_MEMORY = "full-time-no-memory"
    FULL_TIME_NO_BACKCAST = "full-time-no-backcast"
    FULL_TIME_NO_MEMORY_BACKCAST = "full-time-no-memory-backcast"
    NLINEAR_MLSTM = "nLinear+mLSTM+Mem+Back"

    @property
    def no_timemixing(self):
        return self in [
            AblationMode.SLSTM_ONLY,
            AblationMode.SLSTM_MEMORY,
            AblationMode.SLSTM_BACKCASTING,
            AblationMode.SLSTM_MEMORY_BACK,
        ]

    @property
    def no_memory(self):
        return self in [
            AblationMode.NLINEAR_ONLY,
            AblationMode.SLSTM_ONLY,
            AblationMode.SLSTM_BACKCASTING,
            AblationMode.FULL_NO_MEMORY,
            AblationMode.FULL_NO_MEMORY_BACKCAST,
            AblationMode.FULL_TIME_NO_MEMORY,
            AblationMode.FULL_TIME_NO_MEMORY_BACKCAST,
        ]

    @property
    def in_time_mode(self):
        return self in [
            AblationMode.FULL_TIME,
            AblationMode.FULL_TIME_NO_MEMORY,
            AblationMode.FULL_TIME_NO_BACKCAST,
            AblationMode.FULL_TIME_NO_MEMORY_BACKCAST,
        ]

    @property
    def no_backcast(self):
        return self in [
            AblationMode.NLINEAR_ONLY,
            AblationMode.SLSTM_ONLY,
            AblationMode.SLSTM_MEMORY,
            AblationMode.FULL_NO_BACKCAST,
            AblationMode.FULL_NO_MEMORY_BACKCAST,
            AblationMode.FULL_TIME_NO_BACKCAST,
            AblationMode.FULL_TIME_NO_MEMORY_BACKCAST,
        ]


class Model(nn.Module):

    def __init__(
        self,
        configs,
        xlstm_embedding_dim: int = 128,
        num_mem_tokens: int = 0,
        num_tokens_per_variate: int = 1,
        xlstm_dropout: float = 0.1,
        xlstm_conv1d_kernel_size: int = 0,
        xlstm_num_heads: int = 8,
        xlstm_num_blocks: int = 1,
        backcast: bool = True,
        packing: int = 1,
        backbone: Literal["nlinear"] = "nlinear",
        ablation_mode: AblationMode = AblationMode.FULL,
    ) -> None:
        super(Model, self).__init__()
        pred_len = configs.pred_len
        seq_len= configs.seq_len
        enc_in = configs.enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        self.xlstm_embedding_dim = xlstm_embedding_dim

        # self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, seq_len)) if num_mem_tokens > 0 else None
        self.mem_tokens = (
            nn.Parameter(torch.randn(num_mem_tokens, xlstm_embedding_dim) * 0.01)
            if num_mem_tokens > 0
            else None
        )

        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=xlstm_num_heads, conv1d_kernel_size=xlstm_conv1d_kernel_size
            )
        )
        self.ablation_mode = ablation_mode
        self.backcast = backcast

        self.packing = packing

        self.mlp_in = nn.Sequential(
            nn.Linear(self.seq_len, pred_len * num_tokens_per_variate),
        )

        self.mlp_in_trend = nn.Sequential(
            nn.Linear(self.seq_len, pred_len * num_tokens_per_variate),
        )

        if not ablation_mode.no_timemixing:
            self.pre_encoding = nn.Linear(pred_len, xlstm_embedding_dim)
        else:
            self.pre_encoding = nn.Linear(self.seq_len, xlstm_embedding_dim)

        if ablation_mode == AblationMode.FULL_TIME:
            self.xlstm = xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    mlstm_block=(
                        mLSTMBlockConfig()
                        if ablation_mode == AblationMode.NLINEAR_MLSTM
                        else None
                    ),
                    slstm_block=slstm_config,
                    num_blocks=xlstm_num_blocks,
                    embedding_dim=self.enc_in * num_tokens_per_variate + num_mem_tokens,
                    add_post_blocks_norm=True,
                    dropout=xlstm_dropout,
                    bias=True,
                    # slstm_at=[0],
                    slstm_at=(
                        [] if ablation_mode == AblationMode.NLINEAR_MLSTM else "all"
                    ),
                    # slstm_at="all" ,#[0],
                    context_length=xlstm_embedding_dim * self.packing,
                )
            )
        else:
            self.xlstm = xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    mlstm_block=(
                        mLSTMBlockConfig()
                        if ablation_mode == AblationMode.NLINEAR_MLSTM
                        else None
                    ),
                    slstm_block=slstm_config,
                    num_blocks=xlstm_num_blocks,
                    embedding_dim=xlstm_embedding_dim * self.packing,
                    add_post_blocks_norm=True,
                    dropout=xlstm_dropout,
                    bias=True,
                    # slstm_at=[0],
                    slstm_at=(
                        [] if ablation_mode == AblationMode.NLINEAR_MLSTM else "all"
                    ),
                    # slstm_at="all" ,#[0],
                    context_length=self.enc_in * num_tokens_per_variate
                    + num_mem_tokens,  # + 4#336 #self.enc_in #* 2 ,
                )
            )

        if self.backcast:
            self.fc = nn.Linear(self.xlstm_embedding_dim * 2, self.pred_len)
        else:
            self.fc = nn.Linear(self.xlstm_embedding_dim, self.pred_len)

        self.reversible_instance_norm = RevIN(enc_in, affine=False)

        self.decomposition = series_decomp(25)
        self.seq_var_2_var_seq = Rearrange("batch seq var -> batch var seq")
        self.var_seq_2_seq_var = Rearrange("batch var seq -> batch seq var")

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.backbone = backbone

    def forward(self, x_enc, x_mark_enc,_1,_2):
        # norm needs b seq var
        x_enc = self.reversible_instance_norm(x_enc, "norm")

        if self.ablation_mode.no_timemixing:
            x_pre_forecast = self.seq_var_2_var_seq(x_enc)
        elif self.backbone == "nlinear":
            # NLinear
            # x: [Batch, Input length, Channel]
            seq_last = x_enc[:, -1:, :].detach()
            x = x_enc - seq_last
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            x_pre_forecast = x + seq_last
            x_pre_forecast = self.seq_var_2_var_seq(x_pre_forecast)

            if self.ablation_mode == AblationMode.NLINEAR_ONLY:
                x_pre_forecast = self.pre_encoding(x_pre_forecast)
                empty_view = torch.zeros_like(x_pre_forecast)
                x = torch.cat((x_pre_forecast, empty_view), dim=-1)
                x = self.fc(x)
                x = self.var_seq_2_seq_var(x)
                x = self.reversible_instance_norm(x, "denorm")
                return x

        else:
            # Dlinear
            seasonal_init, trend_init = self.decomposition(x_enc)

            seasonal_init = self.seq_var_2_var_seq(seasonal_init)
            trend_init = self.seq_var_2_var_seq(trend_init)

            seasonal_init = self.mlp_in(seasonal_init)
            trend_init = self.mlp_in_trend(trend_init)
            x_pre_forecast = seasonal_init + trend_init

        x = self.pre_encoding(x_pre_forecast)

        if self.packing > 1:
            var = x.shape[1]
            assert (
                var % self.packing == 0
            ), "The number of variables must be divisible by n"

            # Pack variables into sequence
            x = rearrange(x, "b (n var) seq -> b var (seq n)", n=self.packing)

        if self.mem_tokens is not None and not self.ablation_mode.no_memory:
            m: Tensor = repeat(self.mem_tokens, "m d -> b m d", b=x.shape[0])
            x, mem_ps = pack([m, x], "b * d")

        if self.ablation_mode == AblationMode.FULL_TIME:
            x = x.permute(0, 2, 1)

        dim = 1 if self.ablation_mode == AblationMode.FULL_TIME else -1
        if self.backcast and not self.ablation_mode.no_backcast:
            x_reversed = torch.flip(x, [dim])
            x_bwd = self.xlstm(x_reversed)

        x_ = self.xlstm(x)
        x = x_

        if self.backcast:
            if self.ablation_mode.no_backcast:
                x = torch.cat((x, torch.zeros_like(x)), dim=dim)
            else:
                x = torch.cat((x, x_bwd), dim=dim)

        if self.ablation_mode == AblationMode.FULL_TIME:
            x = x.permute(0, 2, 1)

        if self.mem_tokens is not None and not self.ablation_mode.no_memory:
            m, x = unpack(x, mem_ps, "b * d")

        if self.packing > 1:
            x = rearrange(x, "b var (seq n) -> b (var n) seq", n=self.packing)

        x = self.fc(x)

        x = rearrange(x, "b v seq -> b seq v")
        # norm needs b seq var
        x = self.reversible_instance_norm(x, "denorm")

        return x


if __name__ == "__main__":
    class Config:
        def __init__(self):
            self.seq_len = 96
            self.pred_len = 96
            self.enc_in = 7  # 假设变量数量为7

    configs = Config()
    model = xLSTMMixer(configs).cuda()

    batch_size = 4
    seq_len = configs.seq_len
    num_vars = configs.enc_in

    x_enc = torch.randn(batch_size, seq_len, num_vars).cuda()
    x_mark_enc = torch.zeros_like(x_enc)  # 可忽略
    x = model(x_enc, x_mark_enc, None, None)

    print("Output shape:", x.shape)  # 应该是 (batch_size, pred_len, num_vars)
