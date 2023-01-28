'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, hidden_size, num_blocks, strides, num_classes=10, comp_flag=None):
        super(ResNet, self).__init__()
        self.comp_flag = comp_flag
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=strides[2])
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=strides[3])
        self.linear = nn.Linear(hidden_size[3]*block.expansion, num_classes)
        
        
        self.link1 = nn.Sequential()
        if hidden_size[0] != hidden_size[3]:
            self.link1 = nn.Sequential(
                nn.Conv2d(hidden_size[0], hidden_size[3], kernel_size=1, 
                          stride=hidden_size[3]/hidden_size[0], bias=False),
                nn.BatchNorm2d(hidden_size[3])
            )
        
        self.link2 = nn.Sequential()
        if hidden_size[1] != hidden_size[3]:
            self.link2 = nn.Sequential(
                nn.Conv2d(hidden_size[1], hidden_size[3], kernel_size=1, 
                          stride=hidden_size[3]/hidden_size[1], bias=False),
                nn.BatchNorm2d(hidden_size[3])
            )
        
        self.link3 = nn.Sequential()
        if hidden_size[2] != hidden_size[3]:
            self.link3 = nn.Sequential(
                nn.Conv2d(hidden_size[2], hidden_size[3], kernel_size=1, 
                          stride=hidden_size[3]/hidden_size[2], bias=False),
                nn.BatchNorm2d(hidden_size[3])
            )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.comp_flag == 1:
            out = self.layer1(out)
            out = self.link1(out)
        
        elif self.comp_flag == 2:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.link2(out)
            
        elif self.comp_flag == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.link3(out)
            
        elif self.comp_flag == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            
        elif self.comp_flag == 5:
            out = self.link1(out)
            temp_out1 = self.hyper1(out)
            out = self.link2(out)
            temp_out2 = self.hyper2(out)
            out = self.link3(out)
            temp_out3 = self.hyper3(out)
            out = self.link4(out)
            out = out + temp_out1 + temp_out2 + temp_out3
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# def ResNet10():
#     return ResNet(BasicBlock, [64, 64, 64, 64], [1, 1, 1, 1], [1, 2, 2, 2])

def ResNet18(hidden_size, num_blocks, strides, comp_flag):
    return ResNet(BasicBlock, hidden_size=hidden_size, num_blocks=num_blocks, 
                  strides=strides, comp_flag=comp_flag)

# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])

# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])

# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])

# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])
