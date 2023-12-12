from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from .tcn import SingleStageTCN, DilatedResidualLayer
from .SP import MultiScale_GraphConv
from .DeST_linearformer import SFI

    
class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, n_layers, SFI_layer):
        super().__init__()
        self.SFI_layer = SFI_layer
        num_SFI_layers = len(SFI_layer)

        self.conv_in = nn.Conv2d(in_channel, num_SFI_layers+1, kernel_size=1)
        self.conv_t = nn.Conv1d(node, n_features, 1)
        self.SFI_layers = nn.ModuleList(
            [SFI(node, n_features) for i in range(num_SFI_layers)])
        layers = [
            DilatedResidualLayer(2**i, n_features, n_features) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_features, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        count = 0
        x = self.conv_in(x)
        feature_s, feature_t = torch.split(x, (len(self.SFI_layer), 1), dim=1)
        feature_t = feature_t.squeeze(1).permute(0, 2, 1)
        feature_st = self.conv_t(feature_t)

        for index, layer in enumerate(self.layers):
            if index in self.SFI_layer:
                feature_st =  self.SFI_layers[count](feature_s[:,count, :, :], feature_st, mask)
                count+=1
            feature_st = layer(feature_st, mask)

        feature_st = self.conv_out(feature_st)

        return feature_st * mask

class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: N C T V M
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
        self.STI = STI(node, n_features, n_features, n_layers, SFI_layer)
 

        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        # action segmentation branch
        asb = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages_asb - 1)
        ]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_layers) for _ in range(n_stages_brb - 1)
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
                out_cls = as_stage(self.activation_asb(out_cls), mask) * mask[:, 0:1, :]
                outputs_cls.append(out_cls)
            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask) * mask[:, 0:1, :]
                outputs_bound.append(out_bound)
   
            return (outputs_cls, outputs_bound)
        else:
            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls), mask) * mask[:, 0:1, :]
            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask) * mask[:, 0:1, :]

            return (out_cls, out_bound)
