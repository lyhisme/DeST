import torch
import torch.nn as nn
import numpy as np

from .graph.graph import Graph
from .graph.tools import k_adjacency, normalize_adjacency_matrix, get_adjacency_matrix

class MultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,   # 13
                 in_channels,
                 out_channels,
                 dataset,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu'):
        super().__init__()

        self.graph = Graph(labeling_mode='spatial', layout=dataset)
        neighbor = self.graph.neighbor
        self.num_scales = num_scales
        
        if dataset == 'LARA':
            A_binary = get_adjacency_matrix(neighbor, 19)  
        else:
            A_binary = get_adjacency_matrix(neighbor, 25)  
                
        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = torch.Tensor(A_powers)
        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)
    
    def forward(self, x):
        x = x.transpose(2, 3)
        N, C, T, V = x.shape
        self.A_powers = self.A_powers.to(x.device)
        A = self.A_powers.to(x.dtype)
        if self.use_mask:
            A = A + self.A_res.to(x.dtype)
        
        support = torch.einsum('vu,nctu->nctv', A, x)
        support = support.view(N, C, T, self.num_scales, V)
        support = support.permute(0,3,1,2,4).contiguous().view(N, self.num_scales*C, T, V)
        out = self.mlp(support)
        return out


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x

def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)