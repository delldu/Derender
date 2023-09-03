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
from torch import nn
import torch.nn.functional as F
from Derender.networks import netD_model, netA_model, netL_model #, netNR_model
from Derender.renderer import Renderer
from typing import Tuple, Dict
import pdb


def to_gamma_space(img):
    # return img ** GAMMA
    return img**2.2


def from_gamma_space(img):
    # return img.clamp(min=1e-7) ** (1 / GAMMA)
    return img.clamp(min=1e-7) ** (1 / 2.2)


def shrink_mask(mask, shrink: int = 3):
    mask = F.avg_pool2d(mask, kernel_size=shrink, padding=shrink // 2, stride=1)
    return (mask == 1.0).to(torch.float32)


def get_mask(tensor, border: int = 5):
    mask = torch.ones_like(tensor)  # torch .ones(size, dtype=torch.float32)
    mask = shrink_mask(mask, border)
    return mask


class Derender(nn.Module):
    def __init__(self, version="co3d"):
        super().__init__()
        self.min_depth = 0.9
        self.max_depth = 1.1

        self.netD = netD_model(version)
        self.netA = netA_model(version)
        self.netL = netL_model(version)
        # self.netNR = netNR_model(version)
        self.renderer = Renderer()
        self.view_d = self.get_view_d(10.0, 256)
        # self.min_depth = 0.9
        # self.max_depth = 1.1

    def depth_rescaler(self, d):
        return 1.0 + d * 0.1  # (1.0 + d) / 2.0 * self.max_depth + (1.0 - d) / 2.0 * self.min_depth

    def spec_alpha_rescaler(self, x):
        # spec_alpha_max = 64
        # return ((x * .5 + .5) * (math.sqrt(spec_alpha_max) - 1.0) + 1.0) ** 2.0
        return ((x + 1.0) * 3.5 + 1.0) ** 2.0

    def get_view_d(self, fov: float, image_size: int):
        x = torch.linspace(-1, 1, image_size) * math.tan(fov / 360 * math.pi)
        y = x.clone()
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        v = torch.stack([yy, xx, torch.ones_like(xx)], dim=0)
        v = v / torch.norm(v, p=2, dim=0, keepdim=True)
        v = v.flip([1, 2])
        return v.unsqueeze(0)

    # def normal_depth(self, depth):
    #     return (depth - self.min_depth)/(self.max_depth - self.min_depth)


    def predict_shape(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        recon_depth_bump = self.netD(image)
        recon_depth = recon_depth_bump[:, :1, :, :]

        # recon_depth = recon_depth - recon_depth.mean()
        # recon_depth = recon_depth.tanh()
        recon_depth = self.depth_rescaler(recon_depth)

        # B, C, H, W = recon_depth.size()
        # depth_border = torch.zeros((B, C, H, W - 4)).to(recon_depth.device)
        # depth_border = F.pad(depth_border, (2,2), mode='constant', value=1.0)
        # recon_depth = recon_depth * (1.0 - depth_border) + depth_border * recon_depth

        recon_bump = recon_depth_bump[:, 1:, :, :]
        recon_bump = recon_bump / torch.norm(recon_bump, p=2, dim=1, keepdim=True)


        # recon_normal = self.renderer.get_normal_from_depth(recon_depth.squeeze(1)).permute(0, 3, 1, 2)
        recon_normal = self.renderer(recon_depth.squeeze(1)).permute(0, 3, 1, 2)

        recon_normal = recon_normal / torch.norm(recon_normal, p=2, dim=1, keepdim=True)
        recon_normal = recon_normal + recon_bump
        recon_normal = recon_normal / torch.norm(recon_normal, p=2, dim=1, keepdim=True)

        return recon_depth, recon_normal

    def predict_albedo(self, image):
        recon_albedo_specular_notanh = self.netA(image)
        recon_albedo_specular = torch.tanh(recon_albedo_specular_notanh)
        recon_albedo = recon_albedo_specular[:, :3, :, :]

        return recon_albedo

    def forward(self, image) -> Dict[str, torch.Tensor]:
        d: Dict[str, torch.Tensor] = {}

        image = (image - 0.5) * 2.0  # from [0.0, 1.0] to [-1.0, 1.0]

        B = image.shape[0]
        view_d = self.view_d.expand(B, -1, -1, -1).to(image.device)

        recon_depth, recon_normal = self.predict_shape(image)
        d["depth"] = recon_depth #  self.normal_depth(recon_depth)
        d["normal"] = (recon_normal + 1.0)/2.0

        recon_albedo = self.predict_albedo(image)
        d["albedo"] = (recon_albedo + 1.0) / 2.0

        # Light ...
        gamma_recon_albedo = to_gamma_space(recon_albedo / 2.0 + 0.5)

        recon_light_notanh = self.netL(image)
        recon_light = torch.tanh(recon_light_notanh)

        spec_alpha = recon_light_notanh[:, :1].view(B, 1, 1, 1)
        spec_alpha = torch.tanh(spec_alpha)
        spec_alpha = self.spec_alpha_rescaler(spec_alpha)

        spec_strength = (recon_light[:, 5:6].view(B, 1, 1, 1) * 0.5 + 0.5) * 0.5
        spec_strength_min = 0.1
        spec_strength = spec_strength * 2 * (0.5 - spec_strength_min) + spec_strength_min

        recon_light_a = recon_light[:, 1:2] / 2 + 0.5
        recon_light_b = recon_light[:, 2:3] / 2 + 0.5
        recon_light_d = torch.cat([recon_light[:, 3:5], torch.ones(B, 1).to(image.device)], 1)
        recon_light_d = recon_light_d / torch.norm(recon_light_d, p=2, dim=1, keepdim=True)

        cos_theta = (recon_normal * recon_light_d.view(-1, 3, 1, 1)).sum(1, keepdim=True)
        recon_diffuse_shading = cos_theta.clamp(min=0)
        d["diffuse_shading"] = recon_diffuse_shading

        # recon_diffuse_shading.shape -- [1, 1, 256, 256]
        specular_mask = get_mask(recon_diffuse_shading, 5)

        reflect_d = 2 * cos_theta * recon_normal - recon_light_d.view(B, 3, 1, 1)
        specular = (
            (view_d * reflect_d).sum(1, keepdim=True).clamp(min=0) * (cos_theta > 0).to(torch.float32) * specular_mask
        )
        recon_specular_shading = specular.clamp(min=1e-6, max=1.0 - 1e-6).pow(spec_alpha)
        d["specular_shading"] = recon_specular_shading

        recon_shading = recon_light_a.view(-1, 1, 1, 1) + recon_light_b.view(-1, 1, 1, 1) * recon_diffuse_shading
        recon_image = gamma_recon_albedo * recon_shading + recon_specular_shading * spec_strength

        recon_image = from_gamma_space(recon_image)
        recon_image = torch.clamp(recon_image, 0, 1)
        d["image"] = recon_image


        return d
