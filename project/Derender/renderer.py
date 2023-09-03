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
import math
import torch
import torch.nn as nn
import pdb

def get_grid(B: int, H: int, W: int, normalize: bool = True):
    if normalize:
        h_range = torch.linspace(-1.0, 1.0, H)
        w_range = torch.linspace(-1.0, 1.0, W)
    else:
        h_range = torch.arange(0, H)
        w_range = torch.arange(0, W)
    grid = (
        torch.stack(torch.meshgrid([h_range, w_range], indexing="ij"), -1).repeat(B, 1, 1, 1).flip(3).float()
    )  # flip H,W to x,y
    return grid


class Renderer(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 256
        self.fov = 10

        #### camera intrinsics
        #             (u)   (x)
        #    d * K^-1 (v) = (y)
        #             (1)   (z)

        fx = self.image_size / 2 / (math.tan(self.fov / 2 * math.pi / 180))
        fy = self.image_size / 2 / (math.tan(self.fov / 2 * math.pi / 180))
        cx = self.image_size / 2.0
        cy = self.image_size / 2.0
        K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        K = torch.FloatTensor(K)
        self.inv_K = torch.inverse(K).unsqueeze(0)
        # self.K = K.unsqueeze(0)

    def depth_to_3d_grid(self, depth):
        B, H, W = depth.shape
        grid_2d = get_grid(B, H, W, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(self.inv_K.to(depth.device).transpose(2, 1)) * depth
        return grid_3d

    def get_normal_from_depth(self, depth):
        B, H, W = depth.shape
        grid_3d = self.depth_to_3d_grid(depth)

        tu = grid_3d[:, 1:-1, 2:] - grid_3d[:, 1:-1, :-2]
        tv = grid_3d[:, 2:, 1:-1] - grid_3d[:, :-2, 1:-1]
        normal = tu.cross(tv, dim=3)

        zero = torch.FloatTensor([0, 0, 1]).to(depth.device)
        normal = torch.cat([zero.repeat(B, H - 2, 1, 1), normal, zero.repeat(B, H - 2, 1, 1)], 2)
        normal = torch.cat([zero.repeat(B, 1, W, 1), normal, zero.repeat(B, 1, W, 1)], 1)
        normal = normal / (((normal**2).sum(3, keepdim=True)) ** 0.5 + 1e-7)
        return normal

    def forward(self, depth):
        return self.get_normal_from_depth(depth)
