from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import copy
import math

from .tcn import SingleStageTCN
from .SP import MultiScale_GraphConv

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection = nn.Linear(n_features, out_channel)
        self.dropout = nn.Dropout(drop_out)

    def elu(self, x):
        return torch.sigmoid(x)
        # return torch.nn.functional.elu(x) + 1
        
    def forward(self, queries, keys, values, mask):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1) 
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)         
        values = self.value_projection(values).view(B, S, self.n_heads, -1)   
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = self.elu(queries)
        keys = self.elu(keys)
        KV = torch.einsum('...sd,...se->...de', keys, values)  
        Z = 1.0 / torch.einsum('...sd,...d->...s',queries, keys.sum(dim=-2)+1e-6)

        x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1, 2) 
 
        x = x.reshape(B, L, -1) 
        x = self.out_projection(x)
        x = self.dropout(x)

        return x * mask[:, 0, :, None]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
            )
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)
        
        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, x, f, mask):

        out = self.feed_forward(x)
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out
       
        out = self.conv_out(out)
        out = self.dropout(out)

        return (x + out) * mask

class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1) 
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features))
        
    def forward(self, feature_s, feature_t, mask):
        feature_s = feature_s.permute(0, 2, 1)
        n, c, t = feature_s.shape
        feature_s = self.conv_s(feature_s)
        map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t)/t)
        feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t)
        feature_cross = feature_cross + feature_t
        feature_cross = feature_cross.permute(0, 2, 1)
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t

        return feature_cross * mask
    
class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, channel_masking_rate=0.3, alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer
        num_SFI_layers = len(SFI_layer)
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)

        self.conv_in = nn.Conv2d(in_channel, num_SFI_layers+1, kernel_size=1)
        self.conv_t = nn.Conv1d(node, n_features, 1)
        self.SFI_layers = nn.ModuleList(
            [SFI(node, n_features) for i in range(num_SFI_layers)])
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in 
                range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x)

        count = 0
        x = self.conv_in(x)
        feature_s, feature_t = torch.split(x, (len(self.SFI_layers), 1), dim=1)
        feature_t = feature_t.squeeze(1).permute(0, 2, 1)
        feature_st = self.conv_t(feature_t)

        for index, layer in enumerate(self.layers):
            if index in self.SFI_layer:
                feature_st =  self.SFI_layers[count](feature_s[:,count,:], feature_st, mask)
                count+=1
            feature_st = layer(feature_st, None, mask)

        feature_st = self.conv_out(feature_st)
        return feature_st * mask
       
class Decoder(nn.Module):
    def __init__(self, in_channel, n_features, out_channel, num_layers, alpha=1):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'decoder', alpha) for i in 
             range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)
        
        return out, feature

    
class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_refine_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        SFI_layer: Optional[int] = None,
        dataset: str = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()


        self.in_channel = in_channel
        node = 19 if dataset == "LARA" else 25

        self.SP = MultiScale_GraphConv(13, in_channel, n_features, dataset)  
        self.STI = STI(node, n_features, n_features, n_features, n_layers, SFI_layer)
 
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        # action segmentation branch
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.SP(x) * mask.unsqueeze(3)
        feature = self.STI(x, mask)
        
        out_cls = self.conv_cls(feature)
        out_bound = self.conv_bound(feature)
        
        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]

            for as_stage in self.asb:
                out_cls, _ = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)
                outputs_cls.append(out_cls)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound)
        else:
            for as_stage in self.asb:
                out_cls, _ = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)

            return (out_cls, out_bound)
