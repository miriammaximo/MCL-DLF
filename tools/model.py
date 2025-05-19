# Code from MinkUNext repo: https://github.com/juanjo-cabrera/MinkUNeXt.git

import MinkowskiEngine as ME
import torch.nn as nn
import torch
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = self.f(temp)  # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1. / self.p)  # Return (batch_size, n_features) tensor


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                # LayerNorm(planes * block.expansion),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, batch: ME.SparseTensor):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x).F


class ConvNextBlockF(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1):
        super(ConvNextBlockF, self).__init__()
        assert dimension > 0

        # self.dwconv = ME.MinkowskiChannelwiseConvolution(
        #     in_channels=inplanes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.conv = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, dimension=dimension)
        self.norm = LayerNorm(planes, eps=1e-6)
        self.pwconv1 = ME.MinkowskiConvolution(planes, 4 * planes, kernel_size=3, stride=1, dilation=dilation,
                                               dimension=dimension)
        self.norm1 = LayerNorm(4 * planes, eps=1e-6)
        self.pwconv2 = ME.MinkowskiConvolution(4 * planes, planes, kernel_size=3, stride=1, dilation=dilation,
                                               dimension=dimension)
        self.norm2 = LayerNorm(planes, eps=1e-6)
        self.relu = ME.MinkowskiGELU()  # ESTO ERA RELU
        self.downsample = downsample

    def forward(self, x):
        residual = x

        # out = self.dwconv(x)
        #out = self.norm(out)
        #out = self.relu(out) #fewer activations

        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)  # fewer activations

        out = self.pwconv1(out)
        out = self.norm1(out)  # fewer norm
        out = self.relu(out)

        out = self.pwconv2(out)
        out = self.norm2(out)  # fewer norm

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)  # fewer activations
        return out


class MinkUNeXt(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    BLOCK = ConvNextBlockF
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1
    POOLING = None
    FUSION = None
    FINAL_LAYER = 'conv1x1x1'
    add_GFLM = False
    add_1x1x1_convs = False

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3, extract_local = False):
        self.extract_local = extract_local
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])
        if 'BasicBlock' in str(self.BLOCK) or 'ConvNextBlock' in str(self.BLOCK):
            self.final = ME.MinkowskiConvolution(
                128,
                out_channels,
                kernel_size=1,
                bias=True,
                dimension=D)
        elif 'Bottleneck' in str(self.BLOCK):
            self.final = ME.MinkowskiConvolution(
                512,
                out_channels,
                kernel_size=1,
                bias=True,
                dimension=D)

        self.relu1 = ME.MinkowskiReLU(inplace=True)
        self.relu2 = ME.MinkowskiReLU(inplace=True)
        self.relu3 = ME.MinkowskiReLU(inplace=True)
        self.relu4 = ME.MinkowskiReLU(inplace=True)
        self.relu5 = ME.MinkowskiReLU(inplace=True)
        self.relu6 = ME.MinkowskiReLU(inplace=True)
        self.relu7 = ME.MinkowskiReLU(inplace=True)
        self.relu8 = ME.MinkowskiReLU(inplace=True)
        self.GeM_pool = GeM()



    def forward(self, batch):
        # batch = batch[0]
        if self.extract_local is False:
            x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        else:
            x=batch
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu1(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu2(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu3(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu4(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu5(out)
        out = self.block4(out)

        # if self.add_GFLM == True:
        #    out = self.GFLM(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu6(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu7(out)

        out = ME.cat(out, out_b2p4)

        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu8(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        out = self.final(out)
        out = self.GeM_pool(out)

        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, batch):
        x = batch.F
        if self.data_format == "channels_last":
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            out = ME.SparseTensor(x, coordinate_map_key=batch.coordinate_map_key,
                                  coordinate_manager=batch.coordinate_manager)
            # out = ME.SparseTensor(x, batch.C, coordinate_manager=batch.coordinate_manager)
            return out
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            out = ME.SparseTensor(x, batch.C, coordinate_map_key=batch.coordiante_map_key,
                                  coordinate_manager=batch.coordinate_manager)
            return out


