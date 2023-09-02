import math
import torch
import pdb

def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


class Renderer():
    def __init__(self):
        self.image_size = 256
        self.fov = 10

        #### camera intrinsics
        #             (u)   (x)
        #    d * K^-1 (v) = (y)
        #             (1)   (z)

        fx = self.image_size #/2/(math.tan(self.fov/2 *math.pi/180))
        fy = self.image_size/2/(math.tan(self.fov/2 *math.pi/180))
        cx = self.image_size/2
        cy = self.image_size/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K)
        self.inv_K = torch.inverse(K).unsqueeze(0)
        # self.K = K.unsqueeze(0)


    def depth_to_3d_grid(self, depth):
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(self.inv_K.to(depth.device).transpose(2,1)) * depth
        return grid_3d


    def get_normal_from_depth(self, depth):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth)

        tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
        tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
        normal = tu.cross(tv, dim=3)

        zero = torch.FloatTensor([0,0,1]).to(depth.device)
        normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
        normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
        normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + 1e-7)
        return normal
