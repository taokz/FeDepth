# ins --> insufficient memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from .utils.models import ScalableModule

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
hidden_size = [16, 32, 64]


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm_layer, conv_layer, scaler, option='B'):
        super(BasicBlock, self).__init__()
        
        self.bn1 = norm_layer(in_planes)
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.scaler = scaler

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = conv_layer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.scaler(self.shortcut(out)) if hasattr(self, 'shortcut') else x
        out = self.scaler(self.conv1(out))
        out = self.scaler(self.conv2(F.relu(self.bn2(out))))
        out += shortcut
        return out


class ResNet(ScalableModule):

    input_shape = [None, 3, 32, 32]
    
    def __init__(self, hidden_size, block, num_blocks, num_classes=10, bn_type='bn',
                 share_affine=False, track_running_stats=True, width_scale=1.,
                 rescale_init=False, rescale_layer=False, comp_flag=None, alter_flag=None):
        super(ResNet, self).__init__(width_scale=width_scale, rescale_init=rescale_init,
                                     rescale_layer=rescale_layer)
        self.in_planes = 16

        if width_scale != 1.:
            hidden_size = [int(hs * width_scale) for hs in hidden_size]
        self.bn_type = bn_type
        # norm_layer = lambda n_ch: get_bn_layer(bn_type)['2d'](n_ch, track_running_stats=track_running_stats)
        if bn_type == 'bn':
            norm_layer = lambda n_ch: nn.BatchNorm2d(n_ch, track_running_stats=track_running_stats)
        elif bn_type == 'dbn':
            from ..dual_bn import DualNormLayer
            norm_layer = lambda n_ch: DualNormLayer(n_ch, track_running_stats=track_running_stats, affine=True, bn_class=nn.BatchNorm2d,
                 share_affine=share_affine)
        # elif bn_type == 'ln':
        #     norm_layer = lambda n_ch: nn.LayerNorm(n_ch)
        elif bn_type == 'gn':
            norm_layer = lambda n_ch: nn.GroupNorm(4, n_ch) # 3 can be changed -- # of groups
        else:
            raise RuntimeError(f"Not support bn_type={bn_type}")
        conv_layer = nn.Conv2d
        self.first = conv_layer(3, hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.first_norm = norm_layer(hidden_size[0])
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer2 = self._make_layer(block, hidden_size[0], num_blocks[1], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer3 = self._make_layer(block, hidden_size[0], num_blocks[2], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer4 = self._make_layer(block, hidden_size[1], num_blocks[3], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer5 = self._make_layer(block, hidden_size[1], num_blocks[4], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer6 = self._make_layer(block, hidden_size[1], num_blocks[5], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer7 = self._make_layer(block, hidden_size[2], num_blocks[6], stride=2,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer8 = self._make_layer(block, hidden_size[2], num_blocks[7], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.layer9 = self._make_layer(block, hidden_size[2], num_blocks[8], stride=1,
                                       norm_layer=norm_layer, conv_layer=conv_layer)
        self.bn9 = norm_layer(hidden_size[2] * block.expansion)
        self.linear = nn.Linear(hidden_size[2] * block.expansion, num_classes)

        #self.link1 = conv_layer(hidden_size[0], hidden_size[2], kernel_size=1, stride=int(hidden_size[2]/hidden_size[0]), bias=False)
        #self.link2 = conv_layer(hidden_size[0], hidden_size[2], kernel_size=1, stride=int(hidden_size[2]/hidden_size[0]), bias=False)   
        #self.link3 = conv_layer(hidden_size[0], hidden_size[2], kernel_size=1, stride=int(hidden_size[2]/hidden_size[0]), bias=False)
        #self.link4 = conv_layer(hidden_size[1], hidden_size[2], kernel_size=1, stride=int(hidden_size[2]/hidden_size[1]), bias=False)
        #self.link5 = conv_layer(hidden_size[1], hidden_size[2], kernel_size=1, stride=int(hidden_size[2]/hidden_size[1]), bias=False)
        #self.link6 = conv_layer(hidden_size[1], hidden_size[2], kernel_size=1, stride=int(hidden_size[2]/hidden_size[1]), bias=False)
        self.link1 = LambdaLayer(lambda x: F.pad(x[:, :, ::4, ::4], (0, 0, 0, 0, (hidden_size[2]-hidden_size[0])//2, (hidden_size[2]-hidden_size[0])//2), "constant", 0))
        self.link2 = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (hidden_size[2]-hidden_size[1])//2, (hidden_size[2]-hidden_size[1])//2), "constant", 0))
        # for insufficient memory
        self.link3 = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (hidden_size[1]-hidden_size[0])//2, (hidden_size[1]-hidden_size[0])//2), "constant", 0))

        self.reset_parameters(inp_nonscale_layers=['first'])

        self.comp_flag = comp_flag
        self.alter_flag = alter_flag

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, conv_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer, conv_layer, self.scaler))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_pre_clf_fea=False):
        out = self.scaler(self.first_norm(self.first(x)))

        if self.comp_flag == 1:
            if self.alter_flag == 1:
                out = self.layer4(out)
                out = self.link2(out)
            elif self.alter_flag == 2:
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.link2(out)
            elif self.alter_flag == 3:
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.layer6(out)
                out = self.link2(out)
            elif self.alter_flag == 4:
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.layer6(out)
                out = self.layer7(out)
                out = self.layer8(out)
            else:
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.layer6(out)
                out = self.layer7(out)
                out = self.layer8(out)
                out = self.layer9(out)

        elif self.comp_flag == 2:
            if self.alter_flag == 1:
                out = self.layer1(out)
                out = self.link1(out)
            elif self.alter_flag == 2:
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.link1(out)
            elif self.alter_flag == 3:
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.link1(out)
            elif self.alter_flag == 4:
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.link2(out)
            elif self.alter_flag == 5:
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.layer6(out)
                out = self.link2(out)
            else:
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.layer6(out)
                out = self.layer7(out)
                out = self.layer8(out)
                out = self.layer9(out)

        elif self.comp_flag == 3:
            if self.alter_flag == 1:
                # incorrect at the first time
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.link1(out)
                # out = self.layer4(out)
                # out = self.layer5(out)
                # out = self.link2(out)
            else:
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.layer6(out)
                out = self.layer7(out)
                out = self.layer8(out)
                out = self.layer9(out)

        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            out = self.layer7(out)
            out = self.layer8(out)
            out = self.layer9(out)
        
        #print("memory budget flag:", self.comp_flag)
        out = F.relu(self.bn9(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        if return_pre_clf_fea:
            return logits, out
        else:
            return logits

    def print_footprint(self):
        input_shape = self.input_shape
        input_shape[0] = 2
        x = torch.rand(input_shape)
        batch = x.shape[0]
        print(f"input: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        x = self.conv1(x)
        print(f"conv1: {np.prod(x.shape[1:])} <= {x.shape[1:]}")
        for i_layer, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            print(f"layer {i_layer}: {np.prod(x.shape[1:]):5d} <= {x.shape[1:]}")

def init_param(m):
    """Special init for ResNet"""
    if isinstance(m, (_BatchNorm, _InstanceNorm)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

def resnet20(**kwargs):
    model = ResNet(hidden_size, BasicBlock, [1, 1, 1, 1, 1, 1, 1, 1, 1], **kwargs)
    model.apply(init_param)
    return model


# def resnet32(**kwargs):
#     model = ResNet(hidden_size, BasicBlock, [5, 5, 5], **kwargs)
#     model.apply(init_param)
#     return model


# def resnet44(**kwargs):
#     model = ResNet(hidden_size, BasicBlock, [7, 7, 7], **kwargs)
#     model.apply(init_param)
#     return model


# def resnet56(**kwargs):
#     model = ResNet(hidden_size, BasicBlock, [9, 9, 9], **kwargs)
#     model.apply(init_param)
#     return model


# def resnet110(**kwargs):
#     model = ResNet(hidden_size, BasicBlock, [18, 18, 18], **kwargs)
#     model.apply(init_param)
#     return model


# def resnet1202(**kwargs):
#     model = ResNet(hidden_size, BasicBlock, [200, 200, 200], **kwargs)
#     model.apply(init_param)
#     return model


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
