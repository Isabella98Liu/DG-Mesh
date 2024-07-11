import torch
import kiui

from utils.general_utils import build_covariance_from_scaling_rotation, gaussian_3d_coeff


def get_opacity_field_from_gaussians(
    xyzs: torch.Tensor,
    rotations: torch.Tensor,
    scalings: torch.Tensor,
    opacities: torch.Tensor,
    resolution: int = 256,
    num_blocks: int = 16,
    relax_ratio: float = 0.5,
    opacity_threshold: float = 0.005,
    bbox_scale: float = 1.25,
    ) -> torch.Tensor:
    
    block_size = 2 / num_blocks
    assert resolution % block_size == 0
    split_size = resolution // num_blocks

    # pre-filter low opacity gaussians to save computation
    mask = (opacities > opacity_threshold).squeeze(1)

    opacities = opacities[mask]
    rotations = rotations[mask]
    xyzs = xyzs[mask]
    stds = scalings[mask]
    covs = build_covariance_from_scaling_rotation(stds, 1, rotations)

    # tile
    device = opacities.device
    occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

    X = torch.linspace(-bbox_scale, bbox_scale, resolution).split(split_size)
    Y = torch.linspace(-bbox_scale, bbox_scale, resolution).split(split_size)
    Z = torch.linspace(-bbox_scale, bbox_scale, resolution).split(split_size)
    
    # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                # sample points [M, 3]
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                # in-tile gaussians mask
                vmin, vmax = pts.amin(0), pts.amax(0)
                vmin -= block_size * relax_ratio
                vmax += block_size * relax_ratio
                mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                # if hit no gaussian, continue to next block
                if not mask.any():
                    continue
                mask_xyzs = xyzs[mask] # [L, 3]
                mask_covs = covs[mask] # [L, 6]
                mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                # query per point-gaussian pair.
                g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                # batch on gaussian to avoid OOM
                batch_g = 1024
                val = 0
                for start in range(0, g_covs.shape[1], batch_g):
                    end = min(start + batch_g, g_covs.shape[1])
                    w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                    val += (mask_opas[:, start:end] * w).sum(-1)
            
                occ[xi * split_size: xi * split_size + len(xs), 
                    yi * split_size: yi * split_size + len(ys), 
                    zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
    
    kiui.lo(occ, verbose=1)
    
    return occ