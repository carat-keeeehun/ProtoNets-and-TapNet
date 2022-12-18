import torch
import torch.nn as nn
import math

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=True,
                 use_bn=True,
                 max_pool=None,
                 activation="relu"):
        super().__init__()
        self.use_bn = use_bn
        self.use_max_pool = (max_pool is not None)
        self.use_activation = (activation is not None)

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        if self.use_activation:
            self.activation = get_activation(activation)
        if self.use_max_pool:
            self.max_pool = torch.nn.MaxPool2d(max_pool)

    def forward(self, x):
        # convolution layer
        x = self.conv(x)
        # batch normalization
        if self.use_bn:
            x = self.bn(x)
        # ReLU activation
        if self.use_activation:
            x = self.activation(x)
        # 2x2 max pooling
        if self.use_max_pool:
            x = self.max_pool(x)
        return x


def get_activation(activation):
    activation = activation.lower()
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    elif activation == "relu6":
        return torch.nn.ReLU6(inplace=True)
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()

    else:
        raise NotImplementedError("Activation {} not implemented".format(activation))

class ProtoNet(nn.Module):
    '''
    Use 4 block embeddings.
    Each block comprises 64 filter 3x3 convolution, batch normalization, ReLU, and 2x2 max pooling.
    '''
    def __init__(self, input_dim, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.block1 = ConvBlock(input_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block2 = ConvBlock(hid_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block3 = ConvBlock(hid_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block4 = ConvBlock(hid_dim, z_dim, 3, max_pool=2, padding=1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)

        return out


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):
    '''
    Each residual block is constructed with three layers of 3x3 convolutions,
    followed by a batch normalization layer and an ReLU activation function.
    And also have max-pooling layer in the end of residual block.
    '''
    def __init__(self, inplanes, planes, downsample):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.inplanes = 3
        '''
        ResNet12 in TapNet'19 is composed of 4 residual blocks.
        '''
        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes),
            norm_layer(planes),
        )
        block = Block(self.inplanes, planes, downsample)
        self.inplanes = planes
        return block

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x