# Originally written by yabufarha
# https://github.com/yabufarha/ms-tcn/blob/master/model.py

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageTCN(nn.Module):
    """
    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    parameters used in originl paper:
        n_features: 64
        n_stages: 4
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.stage1 = SingleStageTCN(in_channel, n_features, n_classes, n_layers)

        stages = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages - 1)
        ]
        self.stages = nn.ModuleList(stages)

        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            for stage in self.stages:
                out = stage(self.activation(out))
                outputs.append(out)
            return outputs
        else:
            # for evaluation
            out = self.stage1(x)
            for stage in self.stages:
                out = stage(self.activation(out))
            return out

class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out)
        return out * mask[:, 0:1, :]

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)

        return (x + out) * mask[:, 0:1, :]