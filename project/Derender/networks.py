"""Derender Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Sat 02 Sep 2023 07:27:04 PM CST
# ***
# ************************************************************************************/
#
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class PadSameConv2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_y = kernel_size[0]
            self.kernel_size_x = kernel_size[1]
        else:
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor):
        _, _, height, width = x.shape
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) + self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), math.floor(padding_y), math.ceil(padding_y)]
        return F.pad(input=x, pad=padding)


class ConvReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.leaky_relu(t)


class PaddedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return t


class Encoder(nn.Module):
    """
    netL
    """
    def __init__(self, cin, cout, in_size=256, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()

        max_channels = 8 * nf
        num_layers = int(math.log2(in_size)) - 1
        channels = [cin] + [min(nf * (2**i), max_channels) for i in range(num_layers)]

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1 if i != num_layers - 1 else 0,
                        bias=False,
                    ),
                    nn.ReLU(inplace=True),
                )
                for i in range(num_layers)
            ]
        )
        if activation is not None:
            self.out_layer = nn.Sequential(
                nn.Conv2d(max_channels, cout, kernel_size=1, stride=1, padding=0, bias=False), activation()
            )
        else:
            self.out_layer = nn.Conv2d(max_channels, cout, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x).reshape(x.size(0), -1)


class AutoEncoder(nn.Module):
    """
    netA, netNR -- neural_refinement
    """

    def __init__(self, cin=3, cout=4, nf=64, in_size=256, activation=nn.Tanh, depth=9, last_layer_relu=True):
        super().__init__()

        self.max_channels = 16 * nf
        # self.num_layers = min(int(math.log2(in_size)), 5)
        self.num_layers = int(math.log2(in_size))
        if depth is not None:
            self.num_layers = min(depth, self.num_layers)
        self.enc_channels = [cin] + [min(nf * (2**i), self.max_channels) for i in range(self.num_layers)]
        self.dec_channels = [min(nf * (2**i), self.max_channels) for i in reversed(range(self.num_layers))] + [cout]

        self.enc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Identity() if i == 0 else nn.MaxPool2d(2),
                    ConvReLU(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1], kernel_size=3),
                    ConvReLU(
                        in_channels=self.enc_channels[i + 1], out_channels=self.enc_channels[i + 1], kernel_size=3
                    ),
                )
                for i in range(self.num_layers)
            ]
        )
        self.dec = nn.ModuleList(
            [
                nn.Sequential(
                    (ConvReLU if not (i == self.num_layers - 1 and not last_layer_relu) else PaddedConv)(
                        in_channels=(self.dec_channels[i] if i != 0 else 0) + self.enc_channels[-(i + 1)],
                        out_channels=self.dec_channels[i + 1],
                        kernel_size=3,
                    ),
                    (ConvReLU if not (i == self.num_layers - 1) else PaddedConv)(
                        in_channels=self.dec_channels[i + 1], out_channels=self.dec_channels[i + 1], kernel_size=3
                    ),
                    nn.Identity() if i == self.num_layers - 1 else nn.Upsample(scale_factor=2),
                )
                for i in range(self.num_layers)
            ]
        )

        self.predictor = nn.Sequential(activation())

        self.freeze()

    def forward(self, x):
        enc_feats = []
        for layer in self.enc:
            x = layer(x)
            enc_feats.append(x)

        for i, layer in enumerate(self.dec):
            if i == 0:
                x = enc_feats[-1]
            else:
                x = torch.cat([x, enc_feats[-(i + 1)]], dim=1)
            x = layer(x)

        x = self.predictor(x)  # size() -- [1, 4, 256, 256]
        return x  # range [-1.0, 1.0]

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

def load_weights(model, model_path="models/co3d.pth"):
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    if os.path.exists(checkpoint):
        print(f"Loading weight from {checkpoint} ...")
        model.load_state_dict(torch.load(checkpoint))
    else:
        print("-" * 32, "Warnning", "-" * 32)
        print(f"Weight file '{checkpoint}' not exist !!!")


def netD_model(version="co3d"):
    model = AutoEncoder(cin=3, cout=4, nf=64, depth=9)
    load_weights(model, model_path=f"models/{version}_netD.pth")
    return model


def netA_model(version="co3d"):
    model = AutoEncoder(cin=3, cout=5, nf=64, activation=nn.Identity, depth=9)
    load_weights(model, model_path=f"models/{version}_netA.pth")
    return model


def netL_model(version="co3d"):
    """
    Light output:  [ambient strength, directional strength, d_x, d_y, d_z]
    """
    model = Encoder(cin=3, cout=6, nf=32, activation=nn.Identity)

    load_weights(model, model_path=f"models/{version}_netL.pth")
    return model


# def netNR_model(version="co3d"):
#     model = AutoEncoder(cin=3, cout=1, nf=32, depth=6, activation=nn.Sigmoid, last_layer_relu=False)
#     load_weights(model, model_path=f"models/{version}_netNR.pth")
#     return model


if __name__ == "__main__":
    # model = netD_model()
    # model = netA_model()
    model = netL_model()
    # model = netNR_model()

    # model = torch.jit.script(model)

    print(model)
