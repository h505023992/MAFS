import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import numpy as np
from einops.layers.torch import Rearrange

class PatchTST_agent(nn.Module):

    def __init__(self, configs,seq_len,pred_len,patch_len):
        super(PatchTST_agent, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Embedding
        self.d_model = configs.d_model
        self.patch_len = patch_len
        self.patch_num = self.seq_len//patch_len
        self.basis_num = min(self.patch_num,6)
        self.time_base = nn.Linear(self.patch_num, self.basis_num)
        self.enc_embedding = nn.Linear(patch_len, configs.d_model)
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
class AgentGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim,adj):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.adj = adj
    def forward(self, H):
        """
        H: [N, B, C, D] - agent embeddings
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
            self.multi_grained_input = [24,48,72,96]
            self.patch_len = [4,8,12,16]
        elif configs.seq_len == 192:
            self.multi_grained_input = [48,96,144,192]
            self.patch_len = [4,8,12,16]
        elif configs.seq_len == 336:
            self.multi_grained_input = [96,192,288,336]
            self.patch_len = [4,8,12,16]
        elif configs.seq_len == 720:
            self.multi_grained_input = [180,360,540,720]
            self.patch_len = [4,8,12,16]
        
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
            self.agents.append(PatchTST_agent(configs, self.multi_grained_input[i%4], self.multi_grained_input[i%4],self.patch_len[i%4]))
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

    def multi_agent_forecasting(self, x_enc, agent_final_embedding):
        b,t,c = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = (x_enc - means)/stdev
        context_embed = self.context_embed(x_enc.permute(0,2,1))
        new_embeddings = []
        for i in range(self.agent_num):
            # 
            e = agent_final_embedding[i][:,-1,:]
            e = e.view(b,c,-1)
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

        if finetune:
            self.communication.adj = self.learnable_adj()

        multi_grained_x_enc = []
        multi_grained_y_enc = []
        for i in range(4):
            #print(self.multi_grained_input[i])
            multi_grained_x_enc.append(x_enc[:,-self.multi_grained_input[i]:,:])
            multi_grained_y_enc.append(y[:,:self.multi_grained_input[i],:])
        
        for i, agent in enumerate(self.agents):
            #print(multi_grained_x_enc[i%4].shape)
            first_layer_output = agent.initial_embed(multi_grained_x_enc[i%4])
            agent_outputs.append(first_layer_output)
        
   
        #central_output = 0.0
        for layer in range(self.num_layers): 
            second_layer_outputs = []
            for i, agent in enumerate(self.agents):
                # 
                if layer == 0:
                    layer_input = agent_outputs[i]
                else:
                    layer_input = agent_outputs[i] + central_output[i]
                layer_output,_ = agent.encoder_layers[layer](layer_input)   
                if layer  == self.num_layers-1:
                    layer_output = agent.norm(layer_output)
                    #print(layer_output.shape)
                second_layer_outputs.append(layer_output)
   
            central_output = self.communication(second_layer_outputs)
            agent_outputs = second_layer_outputs  


        final_outputs = []
        for i, agent in enumerate(self.agents):
            #print(agent_outputs[i].shape)
            final_output = agent.project(agent_outputs[i])
            #print(final_output.shape)
            final_outputs.append(final_output)
        final_forecasting = self.multi_agent_forecasting(x_enc, agent_outputs)
        if cal_agent_loss:
            agent_loss = sum(self.criterion(final_outputs[i], multi_grained_y_enc[i%4]) for i in range(self.agent_num))
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
            self.agent_num =5
            self.seq_len = 96
            self.pred_len = 96
            self.d_model = 15
            self.enc_in = 7
            self.embed = 'fixed'
            self.freq = 'h'
            self.dropout = 0.1
            self.factor = 3
            self.n_heads = 4
            self.d_ff = 64
            self.e_layers = 2
            self.activation = 'gelu'
            self.mode = 'fully'
            self.patch_len=16

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
