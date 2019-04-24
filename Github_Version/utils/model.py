# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by Chuanting Zhang
** Name: model
** Date: 5/15/18
** Email: chuanting.zhang@gmail.com
** BSD license
********************************************/
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

device = torch.device("cuda")

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell
    """

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.Gates = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                               out_channels=4 * self.hidden_dim,
                               kernel_size=self.kernel_size,
                               padding= self.padding)


    def forward(self, input_tensor, cur_state):


        h_cur, c_cur = cur_state
        h_cur = h_cur.to(device)
        c_cur = c_cur.to(device)
        combined = torch.cat([input_tensor, h_cur], dim=1)

        gates = self.Gates(combined)

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = self.hard_sigmoid(in_gate)
        remember_gate = self.hard_sigmoid(remember_gate)
        out_gate = self.hard_sigmoid(out_gate)

        cell_gate = F.tanh(cell_gate)

        cell = (remember_gate * c_cur) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return hidden, cell

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width))

    def hard_sigmoid(self, x):
        """
        Computes element-wise hard sigmoid of x.
        See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
        """
        x = (0.2 * x) + 0.5
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)

        return x

class ConvLSTMLayer(nn.Module):
    def __init__(self, filters, kernel_size, input_shape, return_sequences=False):
        super(ConvLSTMLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        # the input_shape is 5D tensor (batch, sequences, channels, height, width)
        self.input_shape = input_shape
        self.height = input_shape[3]
        self.width = input_shape[4]
        self.channels = input_shape[2]
        self.sequences = input_shape[1]
        self.return_sequences = return_sequences

        self.CLCell = ConvLSTMCell(input_size=(self.height, self.width),
                                   input_dim=self.channels,
                                   hidden_dim=self.filters,
                                   kernel_size=self.kernel_size)

    def forward(self, x, hidden_state=None):

        if hidden_state is None:
            hidden_state = (
                torch.zeros(x.size(0), self.filters, self.height, self.width),
                torch.zeros(x.size(0), self.filters, self.height, self.width)
            )

        T = x.size(1)
        h, c = hidden_state
        output_inner = []
        for t in range(T):
            h, c = self.CLCell(x[:, t], cur_state=[h, c])
            output_inner.append(h)
        layer_output = torch.stack(output_inner, dim=1)

        if self.return_sequences:
            return layer_output
        else:
            return layer_output[:, -1]



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                            kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1,
                                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, input):
        new_features = super(_DenseLayer, self).forward(input.contiguous())
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([input, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class iLayer(nn.Module):
    def __init__(self):
        super(iLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(1))

    def forward(self, x):
        w = self.w.expand_as(x)
        return x * w


class DenseNet(nn.Module):
    def __init__(self, input_shape, meta_shape, cross_shape, growth_rate=12, num_init_features=16, bn_size=4,
                 drop_rate=0.2, nb_flows=1, fusion=1, maps=1):
        super(DenseNet, self).__init__()
        self.input_shape = input_shape
        self.meta_shape = meta_shape
        self.cross_shape = cross_shape
        self.filters = num_init_features
        self.channels = nb_flows
        self.fusion = fusion
        self.maps = maps
        self.h, self.w = self.input_shape[-2], self.input_shape[-1]
        self.inner_shape = self.input_shape[:2] + (self.filters, ) + self.input_shape[-2:]

        self.temporal = nn.Sequential(OrderedDict([
            ('LSTM0', ConvLSTMLayer(self.filters, 3, input_shape=self.input_shape, return_sequences=True)),
            ('LSTM1', ConvLSTMLayer(self.filters, 3, input_shape=self.inner_shape, return_sequences=True)),
            ('LSTM2', ConvLSTMLayer(self.filters, 3, input_shape=self.inner_shape, return_sequences=False)),
        ]))

        self.meta = nn.Sequential(OrderedDict([
            ('Dense0', nn.Linear(self.meta_shape[1], self.meta_shape[1])),
            ('Activation0', nn.ReLU(inplace=True)),
            ('Dense1', nn.Linear(self.meta_shape[1], self.filters * self.h * self.w))
        ]))

        self.cross = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(4, self.filters, kernel_size=3, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(self.filters)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1, bias=False))
        ]))

        # Each denseblock
        self.features = nn.Sequential()
        num_features = self.filters*maps
        block_config = [6, 6, 6]
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        print(num_features)

        # change from batch norm to relu
        self.features.add_module('relulast', nn.ReLU(inplace=True))
        self.features.add_module('convlast', nn.Conv2d(num_features, nb_flows,
                                                       kernel_size=1, padding=0, bias=False))

        self.iLayer = iLayer()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, meta=None, cross=None):
        out = self.temporal(x)

        if meta is not None:
            meta_out = self.meta(meta)
            meta_out = meta_out.view(-1, self.filters, self.h, self.w)
            out = torch.cat([out, meta_out], dim=1)

        if cross is not None:
            cross_out = self.cross(cross)
            out = torch.cat([out, cross_out], dim=1)



        out = F.sigmoid(self.features(out))
        return out