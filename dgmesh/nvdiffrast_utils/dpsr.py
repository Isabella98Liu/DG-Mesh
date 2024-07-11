# Code from https://github.com/autonomousvision/shape_as_points/blob/main/src/dpsr.py

import torch
import torch.nn as nn
from .dpsr_utils import spec_gaussian_filter, fftfreqs, img, grid_interp, point_rasterize
import numpy as np
import torch.fft

class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        # self.sig = nn.Parameter(torch.tensor(sig, dtype=torch.float32, device='cuda').requires_grad_(True))
        self.dim = len(res)
        self.denom = np.prod(res)
        self.G = spec_gaussian_filter(res=res, sig=self.sig).float()
        # self.G.requires_grad = True # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        # self.register_buffer("G", self.G)
        
    def forward(self, V, N):
        """
        :param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert(V.shape == N.shape) # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]
        
        ras_s = torch.fft.rfftn(ras_p, dim=(2,3,4))
        ras_s = ras_s.permute(*tuple([0]+list(range(2, self.dim+1))+[self.dim+1, 1]))
        
        # G = spec_gaussian_filter(res=self.res, sig=self.sig).float()
        N_ = ras_s[..., None] * self.G.to(V.device) # [b, dim0, dim1, dim2/2+1, n_dim, 1]

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1) # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega *= 2 * np.pi  # normalize frequencies
        omega = omega.to(V.device)
        
        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)
        
        Lap = -torch.sum(omega**2, -2) # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap+1e-6) # [b, dim0, dim1, dim2/2+1, 2]  
        Phi = Phi.permute(*tuple([list(range(1,self.dim+2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b] 
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(*tuple([[self.dim+1] + list(range(self.dim+1))]))  # [b, dim0, dim1, dim2/2+1, 2]
        
        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1,2,3))
        
        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1) # [b, nv]
            if self.shift: # offset points to have mean of 0
                offset = torch.mean(fv, dim=-1)  # [b,] 
                phi -= offset.view(*tuple([-1] + [1] * self.dim))
                
            phi = phi.permute(*tuple([list(range(1,self.dim+1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))
            
            if self.scale:
                phi = -phi / torch.abs(fv0.view(*tuple([-1]+[1] * self.dim))) *0.5
        return phi