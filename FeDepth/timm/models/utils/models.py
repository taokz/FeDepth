import logging
import math
from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.modules.conv import _ConvNd

from .bn_ops import get_bn_layer
from .dual_bn import DualNormLayer


class BaseModule(nn.Module):
    def set_bn_mode(self, is_noised: Union[bool, torch.Tensor]):
        """Set BN mode to be noised or clean. This is only effective for StackedNormLayer
        or DualNormLayer."""
        def set_bn_eval_(m):
            if isinstance(m, (DualNormLayer,)):
                if isinstance(is_noised, (float, int)):
                    m.clean_input = 1. - is_noised
                elif isinstance(is_noised, torch.Tensor):
                    m.clean_input = ~is_noised
                else:
                    m.clean_input = not is_noised
        self.apply(set_bn_eval_)

    # forward
    def forward(self, x):
        z = self.encode(x)
        logits = self.decode_clf(z)
        return logits

    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        return z

    def decode_clf(self, z):
        logits = self.classifier(z)
        return logits

    def mix_dual_forward(self, x, lmbd, deep_mix=False):
        if deep_mix:
            self.set_bn_mode(lmbd)
            logit = self.forward(x)
        else:
            # FIXME this will result in unexpected result for non-dual models?
            logit = 0
            if lmbd < 1:
                self.set_bn_mode(False)
                logit = logit + (1 - lmbd) * self.forward(x)

            if lmbd > 0:
                self.set_bn_mode(True)
                logit = logit + lmbd * self.forward(x)
        return logit


def kaiming_uniform_in_(tensor, a=0, mode='fan_in', scale=1., nonlinearity='leaky_relu'):
    """Modified from torch.nn.init.kaiming_uniform_"""
    fan_in = nn.init._calculate_correct_fan(tensor, mode)
    fan_in *= scale
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def scale_init_param(m, scale_in=1.):
    """Scale w.r.t. input dim."""
    if isinstance(m, (nn.Linear, _ConvNd)):
        kaiming_uniform_in_(m.weight, a=math.sqrt(5), scale=scale_in, mode='fan_in')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            fan_in *= scale_in
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    return m


class Scaler(nn.Module):
    def __init__(self, width_scale):
        super(Scaler, self).__init__()
        self.width_scale = width_scale

    def forward(self, x):
        return x / self.width_scale if self.training else x


class ScalableModule(BaseModule):
    def __init__(self, width_scale=1., rescale_init=False, rescale_layer=False):
        super(ScalableModule, self).__init__()
        if rescale_layer:
            self.scaler = Scaler(width_scale)
        else:
            self.scaler = nn.Identity()
        self.rescale_init = rescale_init
        self.width_scale = width_scale

    def reset_parameters(self, inp_nonscale_layers):
        if self.rescale_init and self.width_scale != 1.:
            for name, m in self._modules.items():
                if name not in inp_nonscale_layers:  # NOTE ignore the layer with non-slimmable inp.
                    m.apply(lambda _m: scale_init_param(_m, scale_in=1./self.width_scale))

    @property
    def rescale_layer(self):
        return not isinstance(self.scaler, nn.Identity)

    @rescale_layer.setter
    def rescale_layer(self, enable=True):
        if enable:
            self.scaler = Scaler(self.width_scale)
        else:
            self.scaler = nn.Identity()
