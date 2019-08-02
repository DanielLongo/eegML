import torch
import torch.nn as nn
import torchvision as vision
import math
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import os
import sys
import inspect
from collections import OrderedDict

# Add parent dir to the sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from constants import *

## Adapted from: https://github.com/gpleiss/efficient_densenet_pytorch

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        #bottleneck_output = conv(relu(norm(concated_features)))
        
        bottleneck_output = relu(norm(conv(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False))
        self.add_module('norm1', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        
        #self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        #self.add_module('relu1', nn.ReLU(inplace=True)),
        #self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
        #                kernel_size=1, stride=1, bias=False)),
        #self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        #self.add_module('relu2', nn.ReLU(inplace=True)),
        #self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                kernel_size=3, stride=1, padding=1, bias=False)),
        
        self.drop_rate = drop_rate
        self.efficient = efficient
    

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        #new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        new_features = self.relu2(self.norm2(self.conv2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))       
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        
        #self.add_module('norm', nn.BatchNorm2d(num_input_features))
        #self.add_module('relu', nn.ReLU(inplace=True))
        #self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
        #                                  kernel_size=1, stride=1, bias=False))
        #self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        
        blk_features = torch.cat(features, 1)
        return blk_features


class SeizureNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        args: args dict
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, args, efficient=False):
        super(SeizureNet, self).__init__()
        assert 0 < args.compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 7

        # First convolution        
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, NUM_INIT_FEATURES, kernel_size=7, stride=2, padding=3, bias=False)),
            ])) # (64, 112, 112)
        self.features.add_module('norm0', nn.BatchNorm2d(NUM_INIT_FEATURES)) # (64, 112, 112)
        self.features.add_module('relu0', nn.ReLU(inplace=True)) # (64, 112, 112)
        self.features.add_module('pool0', nn.AvgPool2d(kernel_size=3, stride=2, padding=1, 
                                                       ceil_mode=False)) # (64, 56, 56)
        

        # Each denseblock
        num_features = NUM_INIT_FEATURES
        for i, num_layers in enumerate(BLOCK_CONFIG):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=BN_SIZE,
                growth_rate=args.growth_rate,
                drop_rate=args.drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * args.growth_rate
            if i != len(BLOCK_CONFIG) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * args.compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * args.compression)
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Linear layer
        self.classifier = nn.Linear(num_features, NUM_CLASSES)

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
                #param.data.normal_(0.0, 0.01) # according to SeizureNet, they initialize with Gaussian of zero-mean and std=0.01               
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)       
        out = F.avg_pool2d(features, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out