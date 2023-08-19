#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init
from collections import OrderedDict

def weight_init(m):
    ''' Initialize weights of a model.
        Borrowed from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class ResNet18(nn.Module):
    """ Model to predict x and y flow from radar heatmaps.
        Based on ResNet18 "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, range_flag=True):
        super(ResNet18, self).__init__()

        self.range_flag = range_flag

        # CNN encoder for heatmaps
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), 
                                        padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(512, 2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, input):
        ranges   = input['range']
        heatmaps = input['radar1']

        heatmaps_enc = self.resnet18(heatmaps)

        if self.training and self.range_flag:
            flow_x = torch.arctan(heatmaps_enc[:,0] / ranges[:,0].clamp(0.1, ))
            flow_y = torch.arctan(heatmaps_enc[:,1] / ranges[:,0].clamp(0.1, ))
            flow = torch.stack((flow_x, flow_y), -1)
        else:
            flow = heatmaps_enc
        return flow

class ResNet18Mini(nn.Module):
    """ Smaller model to predict x and y flow from radar heatmaps.
        Based on ResNet18 "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, range_flag=True):
        super(ResNet18Mini, self).__init__()

        self.range_flag = range_flag

        # CNN encoder for heatmaps
        self.resnet18 = models.resnet._resnet('resnet18', models.resnet.BasicBlock, [1,1,1,1], pretrained=False, progress=False)
        self.resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), 
                                        padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(512, 2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, input):
        ranges   = input['range']
        heatmaps = input['radar1']

        heatmaps_enc = self.resnet18(heatmaps)

        if self.training and self.range_flag:
            flow_x = torch.arctan(heatmaps_enc[:,0] / ranges[:,0].clamp(0.1, ))
            flow_y = torch.arctan(heatmaps_enc[:,1] / ranges[:,0].clamp(0.1, ))
            flow = torch.stack((flow_x, flow_y), -1)
        else:
            flow = heatmaps_enc
        return flow

class ResNet18Micro(nn.Module):
    """ Even smaller model to predict x and y flow from radar heatmaps.
        Based on ResNet18 "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, range_flag=True):
        super(ResNet18Micro, self).__init__()

        self.range_flag = range_flag

        # CNN encoder for48eatmaps
        resnet18 = models.resnet._resnet('resnet18', models.resnet.BasicBlock, [1,1,1,1], pretrained=False, progress=False)
        resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), 
                                   padding=(3, 3), bias=False)
        self.enc = nn.Sequential(OrderedDict(list(resnet18.named_children())[:6]))
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(128, 2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, input):
        ranges   = input['range']
        heatmaps = input['radar1']

        heatmaps_enc = self.enc(heatmaps)
        heatmaps_enc = self.avgpool(heatmaps_enc)
        heatmaps_enc = torch.flatten(heatmaps_enc, 1)
        heatmaps_enc = self.fc(heatmaps_enc)

        if self.training and self.range_flag:
            flow_x = torch.arctan(heatmaps_enc[:,0] / ranges[:,0].clamp(0.1, ))
            flow_y = torch.arctan(heatmaps_enc[:,1] / ranges[:,0].clamp(0.1, ))
            flow = torch.stack((flow_x, flow_y), -1)
        else:
            flow = heatmaps_enc
        return flow

class ResNet18Nano(nn.Module):
    """ Smallest model to predict x and y flow from radar heatmaps.
        Based on ResNet18 "Deep Residual Learning for Image Recognition" https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, range_flag=True):
        super(ResNet18Nano, self).__init__()

        self.range_flag = range_flag

        resnet18 = models.resnet._resnet('resnet18', models.resnet.BasicBlock, [1,1,1,1], pretrained=False, progress=False)
        resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), 
                                   padding=(3, 3), bias=False)
        self.enc = nn.Sequential(OrderedDict(list(resnet18.named_children())[:5]))
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(64, 2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, input):
        ranges   = input['range']
        heatmaps = input['radar1']

        heatmaps_enc = self.enc(heatmaps)
        heatmaps_enc = self.avgpool(heatmaps_enc)
        heatmaps_enc = torch.flatten(heatmaps_enc, 1)
        heatmaps_enc = self.fc(heatmaps_enc)

        if self.training and self.range_flag:
            flow_x = torch.arctan(heatmaps_enc[:,0] / ranges[:,0].clamp(0.1, ))
            flow_y = torch.arctan(heatmaps_enc[:,1] / ranges[:,0].clamp(0.1, ))
            flow = torch.stack((flow_x, flow_y), -1)
        else:
            flow = heatmaps_enc
        return flow

