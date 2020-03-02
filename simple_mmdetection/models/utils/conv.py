from torch import nn


def conv_layer(*args, **kwargs):
    return nn.Conv2d(*args, **kwargs)


def norm_layer(num_features, postfix=''):
    return 'bn' + str(postfix), nn.BatchNorm2d(num_features=num_features)


class ConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 activation=None,
                 ):
        super().__init__()

        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        assert isinstance(activation, nn.Module) or activation is None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
