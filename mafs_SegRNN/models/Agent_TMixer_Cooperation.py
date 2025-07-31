import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from einops.layers.torch import Rearrange

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer_agent(nn.Module):

    def __init__(self, configs,seq_len,pred_len):
        super(TimeMixer_agent, self).__init__()
        self.configs = configs
  
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.encoder_layers = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )


        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence:
            self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        self.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forward(self, x_enc):
        x_mark_enc = None
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
 
        for i, x in zip(range(len(x_list[0])), x_list[0]):
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.encoder_layers[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                #print('fmm | ',enc_out.shape, out_res.shape)
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def initial_embed(self, x_enc):
        x_mark_enc = None
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            #print(x.shape)
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
 
        for i, x in zip(range(len(x_list[0])), x_list[0]):
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        return enc_out_list,x_list

    def project(self, enc_out_list, x_list):
        B, T, N = x_list[0][0].size()
        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

class AgentGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim,adj):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.adj = adj
    def forward(self, H):
        """
        H: [N,B, MT, D] - agent embeddings
        adj: [N, N] - adjacency matrix
        """
        
        H = torch.stack(H, dim=0)
        N, B, C, D = H.shape
        # Graph message passing
        # print(H.shape,self.adj.shape)
        adj = self.adj.to(H.device)  #
        H_new = torch.einsum('ij,jbcd->ibcd', adj, H)
        # H_new = torch.matmul(self.adj, H)  # [N, B, C, D]
        H_new = self.fc(H_new)  # [N, B, C, D]
        return H_new.view(N, B, C, D)
class LearnableAdjacency(nn.Module):
    def __init__(self, N, mode='fully'):
        super().__init__()
        self.N = N
        self.structure_mask = self.build_mask(N, mode)  #
        self.edge_weights = nn.Parameter(torch.rand(N, N))  #

    def build_mask(self, N, mode='fully'):
        A = torch.zeros(N, N)
        if mode == 'fully':
            A.fill_(1)
            A.fill_diagonal_(0)
        elif mode == 'ring':
            for i in range(N):
                A[i, (i+1)%N] = 1
                A[i, (i-1)%N] = 1
        elif mode == 'chain':
            for i in range(N-1):
                A[i, i+1] = 1
        elif mode == 'star':
            for i in range(1, N):
                A[0, i] = 1
                A[i, 0] = 1
        return A  

    def forward(self):
        # 
        A = torch.sigmoid(self.edge_weights) * self.structure_mask.to(self.edge_weights.device)#  
        A = (A + A.T) / 2
        A.fill_diagonal_(0)
        A_hat = A + torch.eye(self.N, device=A.device)

        D = torch.sum(A_hat, dim=1)
        D_hat = torch.diag(D.pow(-0.5))
        D_hat[D_hat == float('inf')] = 0

        norm_A = D_hat @ A_hat @ D_hat
        return norm_A

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.agent_num = configs.agent_num
        self.num_layers = configs.e_layers  # 
        self.mode = configs.mode  # 'central', 'full', 'ring'
        self.agents = nn.ModuleList()

        if configs.seq_len ==96:
            self.multi_grained_output = [24,48,72,96]
        elif configs.seq_len == 192:
            self.multi_grained_output = [48,96,144,192]
        elif configs.seq_len == 336:
            self.multi_grained_output = [96,192,288,336]
        elif configs.seq_len == 720:
            self.multi_grained_output = [180,360,540,720]
  
        self.multi_length = [0]
        mt=0
        for i in range(configs.down_sampling_layers + 1):
            self.multi_length.append(configs.seq_len // (configs.down_sampling_window ** i))
            mt+=configs.seq_len // (configs.down_sampling_window ** i)
        # for i in range(configs.down_sampling_layers + 1):
        #     self.multi_length.append(configs.seq_len // (configs.down_sampling_window ** i))
        #     mt+=configs.seq_len // (configs.down_sampling_window ** i)
       # print(mt)
       # print(self.multi_length)
        self.mixerlineart = nn.Linear(mt,configs.d_model)
        self.mixerlinearc = nn.Linear(configs.d_model,configs.enc_in)
        # 
        self.context_embed = nn.Linear(configs.seq_len, configs.d_model, bias=True)
        self.cal_gate_alpha = nn.Sequential(
            nn.Linear(configs.d_model*2, configs.d_model, bias=True),
            nn.Sigmoid()
        )


        self.cal_collaboration_w = nn.Sequential(
            nn.Linear(configs.enc_in, 1, bias=True),
            Rearrange('b s 1 -> b 1 s'),  # 相当于 permute(0, 2, 1)
            nn.Linear(configs.seq_len, self.agent_num, bias=True),
            nn.Softmax(dim=-1)
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        for i in range(self.agent_num): 
            # 
           # print(i,self.multi_grained_input[i%4])
            self.agents.append(TimeMixer_agent(configs, configs.seq_len, self.multi_grained_output[i%4]))
        self.criterion = nn.MSELoss()
        adj = self.build_adj(N=self.agent_num, mode=self.mode)  # [N, N]
        self.communication = AgentGraphConv(configs.d_model,configs.d_model,adj)
        self.learnable_adj = LearnableAdjacency(N=self.agent_num, mode=self.mode)  
        # print("Agent number: ", self.agent_num)
        # print(self.mode)
        
        # 
    def build_adj(self,N, mode='fully'):
        A = torch.zeros(N, N)
        if mode == 'fully':
            A.fill_(1)
            A.fill_diagonal_(0)
        elif mode == 'ring':
            for i in range(N):
                A[i, (i+1)%N] = 1
                A[i, (i-1)%N] = 1
        elif mode == 'chain':
            for i in range(N-1):
                A[i, i+1] = 1
        elif mode == 'star':
            for i in range(1, N):
                A[0, i] = 1
                A[i, 0] = 1
        adj = A
        I = torch.eye(N, device=A.device)
        A_hat = adj + I
        # D_hat = torch.diag(torch.sum(A_hat, dim=1)) ** -0.5
        # D_hat[D_hat == float('inf')] = 0
        # D_hat = torch.diag_embed(D_hat)

        # norm_A = D_hat @ A_hat @ D_hat  # [N, N]
        D = torch.sum(A_hat, dim=1)  # [N]
        D_hat = torch.diag(D.pow(-0.5))  # [N, N]
        D_hat[D_hat == float('inf')] = 0  # 
        
        norm_A = D_hat @ A_hat @ D_hat  # [N, N]
        return norm_A 
    def mixer_trans(self,agent_final_embedding):

        # self.mixerlineart = nn.Linear(mt,configs.d_model)
        # self.mixerlinearc = nn.Linear(configs.d_model,configs.enc_in)
        atrans = []
        for a in agent_final_embedding:
            # b mt d -> b mt c-> b c mt -> b c d
            a = self.mixerlinearc(a)
            a = a.permute(0,2,1)
            a = self.mixerlineart(a)
            atrans.append(a)
        return atrans
    def multi_agent_forecasting(self, x_enc, agent_final_embedding):
        agent_final_embedding = self.mixer_trans(agent_final_embedding)
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - means)/stdev
        context_embed = self.context_embed(x_enc.permute(0,2,1))
        new_embeddings = []
        for i in range(self.agent_num):
            # 
            e = agent_final_embedding[i]
            context = torch.cat([context_embed, e], dim=-1)
            alpha = self.cal_gate_alpha(context)
            new_embeddings.append(alpha * e + (1 - alpha) * context_embed)
        agent_final_embedding = torch.stack(new_embeddings) # N x B x C x D
        # 
        collaboration_w = self.cal_collaboration_w(x_enc) # B x 1 x N
        # 
        #print(agent_final_embedding.shape,collaboration_w.shape) # torch.Size([4, 8, 7, 64]) torch.Size([8, 1, 4]) -> torch.Size([4, 8, 7, 64])
        collab_w = collaboration_w.permute(2, 0, 1).unsqueeze(-1)  # [A, B, 1, 1]
        #print(collab_w.shape)
        weighted_embedding = agent_final_embedding * collab_w  # [A, B, C, D]
        #print(weighted_embedding.shape)
        z = torch.sum(weighted_embedding, dim=0)  # B x C x D
        #z = torch.sum(collaboration_w.unsqueeze(-1) * agent_final_embedding.permute(1, 0, 2, 3), dim=1)  # B x C x D
        dec_out = self.projection(z).permute(0, 2, 1) # B x C x T -> B x T x C
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * stdev
        dec_out = dec_out + means
        return dec_out

    def forward(self, x_enc,y, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,cal_agent_loss=False,finetune=False):
        agent_outputs = []
        mixer_x_list = []
        if finetune:
            self.communication.adj = self.learnable_adj()

        multi_grained_x_enc = []
        multi_grained_y_enc = []
        for i in range(4):
            #print(self.multi_grained_input[i])
            multi_grained_x_enc.append(x_enc)
            multi_grained_y_enc.append(y[:,:self.multi_grained_output[i],:])
        
        for i, agent in enumerate(self.agents):
            #print(multi_grained_x_enc[i%4].shape)
            first_layer_output,x_list = agent.initial_embed(multi_grained_x_enc[i%4])
            first_layer_output = torch.cat(first_layer_output,dim=1)
            agent_outputs.append(first_layer_output)
            mixer_x_list.append(x_list)
   
        #central_output = 0.0
        for layer in range(self.num_layers): 
            second_layer_outputs = []
            for i, agent in enumerate(self.agents):
                # 
                if layer == 0:
                    layer_input = agent_outputs[i]
                else:
                    
                    layer_input = agent_outputs[i] + central_output[i]
                
                # 在这一步把layer input变成list
                #print(layer_input.shape)
                layer_input_list=[]
                offset=0
                for j in range(len(self.multi_length)-1):
                    layer_input_list.append(layer_input[:,offset+self.multi_length[j]:offset+self.multi_length[j]+self.multi_length[j+1],:])
                    offset+=self.multi_length[j]
                # for xxx in layer_input_list:
                    #print(xxx.shape)
                layer_output = agent.encoder_layers[layer](layer_input_list)
                # 在这一步把layer output变成tensor
                layer_output = torch.cat(layer_output,dim=1)   
                second_layer_outputs.append(layer_output)
   
            central_output = self.communication(second_layer_outputs)
            agent_outputs = second_layer_outputs  


        final_outputs = []
        for i, agent in enumerate(self.agents):
            #print(agent_outputs[i].shape)
            #print('final | ',agent_outputs[i].shape)
            agent_outputs_list = []
            offset=0
            for j in range(len(self.multi_length)-1):
                agent_outputs_list .append(agent_outputs[i][:,offset+self.multi_length[j]:offset+self.multi_length[j]+self.multi_length[j+1],:])
                offset+=self.multi_length[j]           
            
            final_output = agent.project(agent_outputs_list,mixer_x_list[i])
            #print(final_output.shape)
            final_outputs.append(final_output)
        final_forecasting = self.multi_agent_forecasting(x_enc, agent_outputs)
        if cal_agent_loss:
            agent_loss = sum(self.criterion(final_outputs[i], multi_grained_y_enc[i%4]) for i in range(self.agent_num))
            #print(agent_loss)
            return  final_forecasting, agent_loss
        else:
            return final_forecasting,None
    def freeze_agents(self):
        for agent in self.agents:
            for param in agent.parameters():
                param.requires_grad = False



if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.task_name = 'long_term_forecast'
            self.agent_num =4
            self.seq_len = 96
            self.pred_len = 96
            self.label_len = 48
            self.d_model = 64
            self.enc_in = 5
            self.c_out = 5 
            self.factor = 3
            self.n_heads = 4
            self.d_ff = 64
            self.e_layers = 2
            self.activation = 'gelu'
            self.mode = 'fully'
            self.embed = 'timeF'
            self.freq = 'h'
            self.dropout = 0.1
            self.e_layers = 2
            self.down_sampling_layers = 2
            self.down_sampling_window = 2
            self.down_sampling_method = 'avg'  # choose from 'avg', 'max', 'conv'
            self.moving_avg = 25
            self.top_k = 5
            self.use_norm = 1
            self.decomp_method = 'moving_avg'  # or 'dft_decomp'
            self.channel_independence = False  # or True
    configs = Configs()

    B, T, C = 8, configs.seq_len, configs.enc_in
    x_enc = torch.randn(B, T, C)
    y = torch.randn(B, configs.pred_len, C)  


    model = Model(configs)
    model.eval()

    with torch.no_grad():
        output, agent_loss = model(x_enc, y, cal_agent_loss=True)
        print("Final forecasting output shape:", output.shape)
        print("Agent loss:", agent_loss.item())
