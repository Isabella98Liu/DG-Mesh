#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#  created by Isabella Liu (lal005@ucsd.edu) at 2024/02/21 11:29.
#  
#  Gaussian model with DPSR and DiffMC for 3D shape reconstruction, extend to deformable Gaussians.
#  Add anchoring module, given a reconstructed mesh, anchor the mesh face centroids to the deformed Gaussians


import torch
import numpy as np
from torch import nn
import os
import os.path as osp
import trimesh
import open3d as o3d
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import pytorch3d as p3d
from pytorch3d.ops import knn_points
from tqdm import tqdm
import torchgeometry as tgm

from utils.system_utils import mkdir_p
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import build_covariance_from_scaling_rotation, gaussian_3d_coeff
from utils.system_utils import searchForMaxIteration
from utils.mesh_utils import get_opacity_field_from_gaussians

from diso import DiffDMC, DiffMC
from nvdiffrast_utils.dpsr import DPSR

SMALL_NUMBER = 1e-6


class GaussianModelDPSRDynamicAnchor:
    def __init__(self, sh_degree: int, grid_res: int, density_thres: float, dpsr_sig: float):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._normal = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.normal_activation = torch.nn.functional.normalize
        
        # Define DPSR
        self.dpsr_sig = dpsr_sig
        self.grid_res = grid_res
        self.dpsr = DPSR(res=(grid_res, grid_res, grid_res), sig=self.dpsr_sig).to("cuda")  # sig is a learnable parameter
        self.gaussian_center = torch.Tensor([0.0, 0.0, 0.0]).to("cuda")
        self.gaussian_scale = torch.Tensor([1.0]).to("cuda")
        
        # Define DiffMC
        self.diffmc = DiffMC(dtype=torch.float32).to("cuda")
        density_thres = torch.tensor(density_thres, dtype=torch.float32, device="cuda")
        self.density_thres_param = torch.nn.Parameter(density_thres, requires_grad=True)
        
        self.gaussian_param_list = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation", "normal"]

    def upsample_dpsr(self, res):
        self.dpsr = DPSR(res=(res, res, res), sig=self.dpsr_sig).to("cuda")
    
    @torch.no_grad()
    def update_scale_center(self, deform, total_frames=50, gaussian_ratio=1.1, gaussian_center=[0.0, 0.0, 0.0], real=False):
        if not real:
            ratio_all = []
            center_all = []
            for t in range(total_frames):
                N = self.get_xyz.shape[0] 
                time_input = torch.ones(N, 1, device='cuda') * t / total_frames
                d_xyz_s, _, _, _ = deform.step(self.get_xyz.detach(), time_input)
                points = self.get_xyz + d_xyz_s
                
                center = (torch.max(points.detach(), dim=0).values + torch.min(points.detach(), dim=0).values) / 2.0
                center_all.append(center)
                
                # Find center and ratio to scale the gaussian, only for synthetic data because real points will have floaters
                ratio = torch.max(points.detach(), dim=0).values - torch.min(points.detach(), dim=0).values
                ratio = torch.max(ratio, dim=0).values
                ratio_all.append(ratio)
        
            gaussian_center = torch.mean(torch.stack(center_all), dim=0)
            self.gaussian_center = gaussian_center
            ratio = torch.max(torch.stack(ratio_all), dim=0).values
            gaussian_scale = ratio * gaussian_ratio / 2.0
            # gaussian_scale = ratio * dataset.gaussian_ratio
            self.gaussian_scale = gaussian_scale
        else:
            self.gaussian_scale = torch.tensor([gaussian_ratio]).to("cuda") / 2.0
            self.gaussian_center = torch.tensor(gaussian_center).to("cuda")
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_normal(self):
        return self._normal

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        if np.any(pcd.normals):
            normals = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        else:
            normals = torch.rand((fused_point_cloud.shape[0], 3), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normals.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._normal], 'lr': training_args.rotation_lr * 100, "name": "normal"},
            {'params': [self.density_thres_param], 'lr': 0.01, "name": "density_thres"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.density_thres_scheduler_args = get_expon_lr_func(lr_init=0.01, 
                                                              lr_final=0.0001, 
                                                              lr_delay_mult=0.01, 
                                                              max_steps=training_args.position_lr_max_steps)
        self.normal_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr, 
                                                       lr_final=training_args.rotation_lr * 0.1, 
                                                       lr_delay_mult=0.01, 
                                                       max_steps=training_args.position_lr_max_steps)
        self.rotation_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr * 100, 
                                                        lr_final=training_args.rotation_lr * 100 * 0.1, 
                                                        lr_delay_mult=0.01, 
                                                        max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "density_thres":
                lr = self.density_thres_scheduler_args(iteration)
                param_group['lr'] = lr            
            elif param_group["name"] == "normal":
                lr = self.normal_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # Save Gaussian attributes
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        normals = self._normal.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        
        # Save density threshold and other parameters
        density_thres_np = np.array([self.density_thres_param.detach().cpu()])
        dens_thres = np.empty(1, dtype=[('density_thres', 'f4')])
        dens_thres[:] = list(map(tuple, [density_thres_np]))
        
        gaussian_center_np = self.gaussian_center.view(-1).detach().cpu().numpy()
        gaussian_center = np.empty(1, dtype=[('gaussian_center_x', 'f4'), ('gaussian_center_y', 'f4'), ('gaussian_center_z', 'f4')])
        gaussian_center[:] = list(map(tuple, [gaussian_center_np]))
        
        gaussian_scale_np = np.array([self.gaussian_scale.view(-1).detach().cpu().numpy()])
        gaussian_scale = np.empty(1, dtype=[('gaussian_scale', 'f4')])
        gaussian_scale[:] = list(map(tuple, [gaussian_scale_np]))

        dens_thres_el = PlyElement.describe(dens_thres, 'density_thres')
        center_el = PlyElement.describe(gaussian_center, 'gaussian_center')
        scale_el = PlyElement.describe(gaussian_scale, 'gaussian_scale')
        
        PlyData([el, dens_thres_el, center_el, scale_el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, og_number_points=-1, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(path, "point_cloud"))
        else:
            loaded_iter = iteration
        path = os.path.join(path, "point_cloud/iteration_{}/point_cloud.ply".format(loaded_iter))
        
        self.og_number_points = og_number_points
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        density_thres = np.asarray(plydata.elements[1]["density_thres"])
        gaussian_center = np.stack((np.asarray(plydata.elements[2]["gaussian_center_x"]),
                                    np.asarray(plydata.elements[2]["gaussian_center_y"]),
                                    np.asarray(plydata.elements[2]["gaussian_center_z"])), axis=1)
        gaussian_scale = np.asarray(plydata.elements[3]["gaussian_scale"])
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self.density_thres_param = torch.nn.Parameter(torch.tensor(density_thres, dtype=torch.float32, device="cuda"), requires_grad=True)
        self.gaussian_center = torch.tensor(gaussian_center, dtype=torch.float32, device="cuda")
        self.gaussian_scale = torch.tensor(gaussian_scale, dtype=torch.float32, device="cuda")
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree
        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self.gaussian_param_list:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self.gaussian_param_list:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self.gaussian_param_list:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_normal):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "normal": new_normal}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._normal = optimizable_tensors["normal"]
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normal)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_normal)

    def prune_old(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            small_points_ws = self.get_scaling.max(dim=1).values < 0.001 * extent
            prune_mask = torch.logical_or(prune_mask, small_points_ws)
        self.prune_points(prune_mask)
        
        torch.cuda.empty_cache()
        
    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        new_scaling = self.get_scaling
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = new_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            # small_points_ws = new_scaling.max(dim=1).values < 0.0002 * extent
            # prune_mask = torch.logical_or(prune_mask, small_points_ws)
        self.prune_points(prune_mask)
        
        torch.cuda.empty_cache()
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        self.prune(min_opacity, extent, max_screen_size)

        torch.cuda.empty_cache()

    def average_and_prune_single(self, selected_pts_mask, deform, deform_back, t):
        # Create new gaussian by averaging the selected points under deformed frame
        selected_xyz = self._xyz[selected_pts_mask]  # [N, 3]
        selected_scaling = self._scaling[selected_pts_mask]
        selected_rotation = self._rotation[selected_pts_mask]
        selected_normal = self._normal[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask].mean(0, keepdim=True)  # [1, 1, 3]
        new_features_rest = self._features_rest[selected_pts_mask].mean(0, keepdim=True)  # [1, (self.max_sh_degree + 1) ** 2 - 1, 3]
        new_opacity = self._opacity[selected_pts_mask].mean(0, keepdim=True)
        
        # Deform the selected points in canonical space to the deformed space
        with torch.no_grad():
            time_input = torch.ones(selected_xyz.shape[0], 1, device='cuda') * t
            d_xyz, d_rotation, d_scaling, d_normal  = deform.step(selected_xyz, time_input)
            selected_xyz = selected_xyz + d_xyz
            selected_scaling = selected_scaling + d_scaling
            selected_rotation = selected_rotation + d_rotation
            selected_normal = selected_normal + d_normal
        
        new_xyz = selected_xyz.mean(0, keepdim=True)  # [1, 3]
        deformed_xyz = new_xyz
        new_scaling = selected_scaling.mean(0, keepdim=True)
        new_rotation = selected_rotation.mean(0, keepdim=True)
        new_normal = selected_normal.mean(0, keepdim=True)
        
        # Defom the new gaussian back to canonical space
        with torch.no_grad():
            time_input = torch.ones(new_xyz.shape[0], 1, device='cuda') * t
            d_xyz, d_rotation, d_scaling, d_normal  = deform_back.step(new_xyz, time_input)
            new_xyz = new_xyz + d_xyz
            new_scaling = new_scaling + d_scaling
            new_rotation = new_rotation + d_rotation
            new_normal = new_normal + d_normal
            new_normal = nn.functional.normalize(new_normal, p=2, dim=-1)
        
        # Prune the selected points
        self.prune_points(selected_pts_mask)
        
        # Add the new gaussian
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normal)
        
        torch.cuda.empty_cache()
        
        return deformed_xyz

    def average_and_prune(self, selected_pts_mask, deform, deform_back, t, topn=2):
        # selected_pts_mask [X, G, 1] with G = number of gaussians, X = number of implementing faces, each row has at most 2 gaussians selected
        
        # Create new gaussian by averaging the selected points under deformed frame
        selected_xyz = torch.masked_select(self._xyz.unsqueeze(0), selected_pts_mask).view(-1, topn, self._xyz.shape[-1])  # [bs, 2, 3]
        selected_scaling = torch.masked_select(self._scaling.unsqueeze(0), selected_pts_mask).view(-1, topn, self._scaling.shape[-1])  # [bs, 2, 3]
        selected_rotation = torch.masked_select(self._rotation.unsqueeze(0), selected_pts_mask).view(-1, topn, self._rotation.shape[-1])  # [bs, 2, 4]
        selected_normal = torch.masked_select(self._normal.unsqueeze(0), selected_pts_mask).view(-1, topn, self._normal.shape[-1])  # [bs, 2, 3]
        
        new_features_dc = torch.masked_select(self._features_dc.unsqueeze(0), selected_pts_mask.unsqueeze(-1)).view(-1, topn, self._features_dc.shape[-2], self._features_dc.shape[-1]).mean(1, keepdim=True)  # [bs, 1, 1, 3]
        new_features_rest = torch.masked_select(self._features_rest.unsqueeze(0), selected_pts_mask.unsqueeze(-1)).view(-1, topn, self._features_rest.shape[-2], self._features_rest.shape[-1]).mean(1, keepdim=True)  # [bs, 1, (self.max_sh_degree + 1) ** 2 - 1, 3]

        new_opacity = torch.masked_select(self._opacity.unsqueeze(0), selected_pts_mask).view(-1, topn, self._opacity.shape[-1]).mean(1, keepdim=True)  # [bs, 1, 1]
        
        # Deform the selected points in canonical space to the deformed space
        total_num = selected_xyz.shape[0] * topn
        bs = selected_xyz.shape[0]
        with torch.no_grad():
            time_input = torch.ones(total_num, 1, device='cuda') * t
            d_xyz, d_rotation, d_scaling, d_normal  = deform.step(selected_xyz.view(-1, 3), time_input)
            selected_xyz = selected_xyz + d_xyz.view(bs, topn, -1)
            selected_scaling = selected_scaling + d_scaling.view(bs, topn, -1)
            selected_rotation = selected_rotation + d_rotation.view(bs, topn, -1)
            selected_normal = selected_normal + d_normal.view(bs, topn, -1)
        
        new_xyz = selected_xyz.mean(1, keepdim=True)  # [bs, 1, 3]
        deformed_xyz = new_xyz
        new_scaling = selected_scaling.mean(1, keepdim=True)  # [bs, 1, 3]
        new_rotation = selected_rotation.mean(1, keepdim=True)  # [bs, 1, 4]
        new_normal = selected_normal.mean(1, keepdim=True)  # [bs, 1, 3]
        
        # Defom the new gaussian back to canonical space
        with torch.no_grad():
            time_input = torch.ones(bs, 1, device='cuda') * t
            d_xyz, d_rotation, d_scaling, d_normal  = deform_back.step(new_xyz.view(-1, 3), time_input)
            new_xyz = new_xyz.view(bs, -1) + d_xyz
            new_scaling = new_scaling.view(bs, -1) + d_scaling
            new_rotation = new_rotation.view(bs, -1) + d_rotation
            new_normal = new_normal.view(bs, -1) + d_normal
            new_normal = nn.functional.normalize(new_normal, p=2, dim=-1) # [bs,  3]
        
        # Prune the selected points
        selected_pts_mask_sum = selected_pts_mask.sum(0).view(-1).to(torch.bool)
        self.prune_points(selected_pts_mask_sum)
        
        # Add the new gaussian
        self.densification_postfix(new_xyz, new_features_dc.squeeze(1), new_features_rest.squeeze(1), new_opacity.squeeze(1), new_scaling, new_rotation, new_normal)
        
        torch.cuda.empty_cache()
        
        return deformed_xyz.view(-1, 3)
    
    def densify_from_face(self, new_xyz, new_normal, scale, deform_back, t):
        
        new_xyz = new_xyz.view(-1, 3)
        new_features_dc = torch.ones((new_xyz.shape[0], 1, 3)).float().to(new_xyz.device)
        new_features_rest = torch.zeros((new_xyz.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3)).float().to(new_xyz.device)
        new_opacity = inverse_sigmoid(0.1 * torch.ones((new_xyz.shape[0], 1), dtype=torch.float, device=new_xyz.device))
        dist2 = torch.clamp_min(distCUDA2(new_xyz.float().cuda()), 0.0000001)
        new_scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # new_scaling = torch.ones((new_xyz.shape[0], 3), dtype=torch.float, device=new_xyz.device) * scale
        axis = nn.functional.normalize(new_normal, p=2, dim=-1)
        angle = torch.randn((new_normal.shape[0], 1), device=new_normal.device) * 2 * np.pi
        axis_angle = axis * angle
        new_rotation = p3d.transforms.axis_angle_to_quaternion(axis_angle)
        new_normal = new_normal.view(-1, 3)
        
        # Deform the new gaussian back to canonical space
        with torch.no_grad():
            time_input = torch.ones(new_xyz.shape[0], 1, device='cuda') * t
            d_xyz, d_rotation, d_scaling, d_normal  = deform_back.step(new_xyz, time_input)
            new_xyz = new_xyz + d_xyz
            new_scaling = new_scaling + d_scaling
            new_rotation = new_rotation + d_rotation
            new_normal = new_normal + d_normal
            new_normal = nn.functional.normalize(new_normal, p=2, dim=-1)
    
        # Add the new gaussian
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normal)
        
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def normal_initialization(self, args, opt, dataset, deform, d_xyz, d_rotation, d_scaling):
        if args.data_type == 'iPhone' or args.data_type == 'NeuralActor':
            self.update_scale_center(deform=deform, gaussian_ratio=dataset.gaussian_ratio, gaussian_center=dataset.gaussian_center, real=True)
        else:
            self.update_scale_center(deform=deform, gaussian_ratio=dataset.gaussian_ratio, gaussian_center=dataset.gaussian_center, real=False)
        print(f'Update scale and center, new scale {self.gaussian_scale}, new center {self.gaussian_center}')
        
        # Obtain the original mesh from the gaussian density grid
        occ_bbox_scale = 2.0
        gaussian_xyzs = self.get_xyz + d_xyz
        occ = get_opacity_field_from_gaussians(
            gaussian_xyzs,
            self.get_rotation + d_rotation, 
            self.get_scaling + d_scaling,
            self.get_opacity,
            bbox_scale = occ_bbox_scale
        )
        
        # Extract the mesh 
        verts, faces = self.diffmc(-occ, deform=None, isovalue=-0.01)
        # Scale back
        verts = verts * 2.0 * occ_bbox_scale - occ_bbox_scale  # [-occ_bbox_scale, occ_bbox_scale]
        
        # Query the nearets points' to the mesh faces normal as normal initialization
        ## Sample points on the mesh
        resample_num = self.get_xyz.shape[0]
        occ_mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
        occ_mesh.export(osp.join(args.model_path, 'mesh_init.ply'))
        sampled_points, face_indices = trimesh.sample.sample_surface(occ_mesh, resample_num)
        sampled_points = torch.tensor(sampled_points, dtype=torch.float32, device=gaussian_xyzs.device)
        sampled_normals = occ_mesh.face_normals[face_indices]
        sampled_normals = torch.tensor(sampled_normals, dtype=torch.float32, device=gaussian_xyzs.device)
        
        # Query the nearest mesh surface points to the gaussian points
        _, mesh_idx, _ = knn_points(gaussian_xyzs.unsqueeze(0), sampled_points.unsqueeze(0), K=1)
        gs_normals = sampled_normals[mesh_idx.squeeze()]
        
        pcds = o3d.geometry.PointCloud()
        points = self.get_xyz + d_xyz
        pcds.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        pcds.normals = o3d.utility.Vector3dVector(gs_normals.detach().cpu().numpy())
        
        normals = torch.tensor(gs_normals, dtype=torch.float32, device="cuda")
        self._normal.data = normals
        o3d.io.write_point_cloud(osp.join(args.model_path, 'pointcloud_init.ply'), pcds)
        print("DPSR normal initialization finished")
        
        # Initialize density threshold
        init_thres = opt.init_density_threshold
        self.density_thres_param.data = torch.tensor([init_thres], device='cuda').requires_grad_(True)
    
    def anchor_mesh(self, verts, faces, deform, deform_back, t, search_radius=0.0005, topn=2, bs=256, increase_bs=1024):
        # Anchor gaussians to the mesh
        # Assign each point to the closest face centroid:
        #   1. If multiple gaussians are assigned to the same face, average their centroids and create a new gaussian, delete the old gaussians
        #   2. If a face is not assigned to any gaussians, create new gaussians at the face centroids
        
        torch.cuda.empty_cache()
        search_radius = self.gaussian_scale * search_radius
        
        old_xyz_num = self.get_xyz.shape[0]
    
        # Deform canonical gaussian
        time_input = torch.ones(self.get_xyz.shape[0], 1, device='cuda') * t
        d_xyz, _, _, _ = deform.step(self.get_xyz.detach(), time_input)

        # For each face centroid, find the closest gaussian point
        mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
        face_centroids = mesh.triangles_center  # [F, 3]
        face_normals = mesh.face_normals  # [F, 3]
        face_centroids_tensor = torch.tensor(face_centroids, dtype=torch.float, device=verts.device, requires_grad=False)
        face_normals_tensor = torch.tensor(face_normals, dtype=torch.float, device=verts.device, requires_grad=False)
        gaussian_points = self.get_xyz + d_xyz  # [G, 3]
        
        # For each gaussian point, find the closest face centroid
        gs_face_dist, face_indices, _ = knn_points(gaussian_points.unsqueeze(0), face_centroids_tensor.unsqueeze(0), K=1)
        gs_face_dist = gs_face_dist.squeeze(0)  # [G, 1]
        face_indices = face_indices.squeeze(0)  # [G, 1]
        valid_mask = gs_face_dist < search_radius
        ## Prune the invalid gaussians
        self.prune_points(~valid_mask.squeeze(1))
        invalid_ratio = (~valid_mask).sum().item() / face_indices.shape[0]
        gs_face_dist = gs_face_dist[valid_mask].unsqueeze(1)
        face_indices = face_indices[valid_mask].unsqueeze(1)
        ##########################################
        # Separate the face index into three sets, 
        # 1. have 1-1 gs-face correspondence, 
        # 2. have n-1 gs-face correspondence, 
        # 3. have 0-1 gs-face correspondence
        ##########################################
        # Find the unique face indices
        unique_indices, counts = torch.unique(face_indices, return_counts=True)
        
        face_indices_1_1 = unique_indices[counts == 1]
        face_indices_n_1 = unique_indices[counts > 1]
        face_indices_all = torch.arange(face_centroids_tensor.shape[0], device=gaussian_points.device)
        face_indices_0_1 = face_indices_all[~(torch.isin(face_indices_all, face_indices_1_1) + torch.isin(face_indices_all, face_indices_n_1))]
        assert face_indices_1_1.shape[0] + face_indices_n_1.shape[0] + face_indices_0_1.shape[0] == face_centroids_tensor.shape[0]
        
        # For 1-1 mapped gs-face, calculate the distance loss
        gs_indices_1_1_mask = torch.isin(face_indices, face_indices_1_1)  # [G, 1]
        anchor_loss_1_1 = gs_face_dist[gs_indices_1_1_mask].mean()
        
        # For n-1 mapped gs-face, delete the original gaussians and create new gaussians at the mean position
        ## Randomly select
        random_indices = torch.randperm(face_indices_n_1.shape[0], device=face_indices_n_1.device)[:bs]
        face_indices_n_1 = face_indices_n_1[random_indices]
        face_indices_n_1_expand = face_indices_n_1.view(-1, 1, 1)  # [X, 1, 1]
        face_indices_expand = face_indices.view(1, -1, 1)  # [1, G, 1]
        match_mask = torch.eq(face_indices_n_1_expand, face_indices_expand)  # [X, G, 1]
        match_mask_row_cumsums = torch.cumsum(match_mask, dim=1)  # [X, G, 1]
        match_mask_row_topn = match_mask_row_cumsums <= topn  # [X, G, 1]
        match_mask_topn = torch.logical_and(match_mask, match_mask_row_topn)  # [X, G, 1] with each row has at most topn True
        ## Delete the rest gaussian which is not in the topn
        match_mask_topn_sum = match_mask_topn.sum(0)  # [G, 1]
        match_mask_sum = match_mask.sum(0)  # [G, 1]
        to_delete_mask = torch.logical_xor(match_mask_topn_sum, match_mask_sum)  # [G, 1]
        self.prune_points(to_delete_mask.squeeze(1))
        # Update the mask
        match_mask_topn = torch.masked_select(match_mask_topn, ~to_delete_mask.unsqueeze(0)).view(match_mask_topn.shape[0], -1, 1)  # [X, G, 1]
        new_xyz = self.average_and_prune(match_mask_topn, deform, deform_back, t, topn)  # [bs, 3]
        face_xyz = face_centroids_tensor[face_indices_n_1]
        anchor_loss_n_1 = torch.norm(face_xyz - new_xyz, dim=-1).mean()
        
        # For 0-1 mapped gs-face, create new gaussians at the face centroids
        face_indices_0_1_mask = ~(torch.isin(face_indices_all, face_indices_1_1) + torch.isin(face_indices_all, face_indices_n_1))  # [F, 1]
        face_centroids_tensor_0_1 = face_centroids_tensor[face_indices_0_1_mask]
        face_normals_tensor_0_1 = face_normals_tensor[face_indices_0_1_mask]
        avg_edge_length = mesh.edges_unique_length.mean()
        # Select the face centroids in batchwise
        random_indices = torch.randperm(face_centroids_tensor_0_1.shape[0], device=face_centroids_tensor_0_1.device)[:increase_bs]
        face_centroids_tensor_0_1_batch = face_centroids_tensor_0_1[random_indices]
        face_normals_tensor_0_1_batch = face_normals_tensor_0_1[random_indices]
        self.densify_from_face(face_centroids_tensor_0_1_batch, face_normals_tensor_0_1_batch, avg_edge_length/2, deform_back, t)
        
        anchor_loss = anchor_loss_1_1 + anchor_loss_n_1
        
        new_xyz_num = self.get_xyz.shape[0]
        
        print(f"Old number of gaussians: {old_xyz_num}, New number of gaussians: {new_xyz_num}, Target face number {face_centroids_tensor.shape[0]}, Anchor loss: {anchor_loss:04f} 1-1 Hit rate: {face_indices_1_1.shape[0]/face_centroids_tensor.shape[0]:.4f} Invalid ratio: {invalid_ratio:.4f}")
        
        torch.cuda.empty_cache()
        
        return anchor_loss
    
    @torch.no_grad()
    def export_mesh(self, deform, deform_normal, t=0.0):
        N = self.get_xyz.shape[0]
        time_input = torch.ones(N, 1, device='cuda') * t
        d_xyz, d_rotation, d_scaling, _  = deform.step(self.get_xyz, time_input)
        d_normal = deform_normal.step(self.get_xyz, time_input)
        dpsr_points = self.get_xyz + d_xyz
        dpsr_points = (dpsr_points - self.gaussian_center) / self.gaussian_scale  # [-1, 1]
        dpsr_points = dpsr_points / 2.0 + 0.5  # [0, 1]
        dpsr_points = torch.clamp(dpsr_points, SMALL_NUMBER, 1-SMALL_NUMBER)
        
        normals = self.get_normal + d_normal
        
        ## Query SDF
        psr = self.dpsr(dpsr_points.unsqueeze(0), normals.unsqueeze(0))
        sign = psr[0, 0, 0, 0].detach()  # Sign for Diso is opposite to dpsr
        sign = -1 if sign < 0 else 1
        psr = psr * sign
        psr -= self.density_thres_param
        psr = psr.squeeze(0)
        ## Differentiable Marching Cubes
        verts, faces = self.diffmc(psr, deform=None, isovalue=0.0)
        verts = verts * 2.0 - 1.0  # [-1, 1]
        verts = verts * self.gaussian_scale + self.gaussian_center
        verts = verts.to(torch.float32)
        faces = faces.to(torch.int32)
        return verts, faces

    @torch.no_grad()
    def export_pointcloud(self, deform, deform_normal, t=0.0, d_xyz=None, d_normal=None):
        if d_xyz is not None and d_normal is not None:
            dpsr_points = self.get_xyz + d_xyz
            normals = self.get_normal + d_normal
            normals = torch.nn.functional.normalize(normals, dim=1)
        else:
            N = self.get_xyz.shape[0]
            time_input = torch.ones(N, 1, device='cuda') * t
            d_xyz, d_rotation, d_scaling, _  = deform.step(self.get_xyz, time_input)
            d_normal = deform_normal.step(self.get_xyz, time_input)
            
            normals = self.get_normal + d_normal
            
            dpsr_points = self.get_xyz + d_xyz

        return dpsr_points, normals