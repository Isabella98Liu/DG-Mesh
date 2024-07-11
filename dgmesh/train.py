#  created by Isabella Liu (lal005@ucsd.edu) at 2024/05/28 16:10.
#
#  Training script for DG-Mesh


import os, sys
import json
import datetime
import os.path as osp
import torch
import uuid
import datetime
from tqdm import tqdm
import random
from argparse import ArgumentParser, Namespace
import numpy as np
import imageio
import trimesh
from pytorch_msssim import ms_ssim as MS_SSIM
import open3d as o3d
from pytorch3d.ops import knn_points
import nvdiffrast.torch as dr

from scene import Scene
from scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from scene import DeformModelNormal as deform_model
from scene import DeformModelNormalSep as deform_model_sep
from scene import AppearanceModel as appearance_model
from utils.renderer import mesh_renderer
from nvdiffrast_utils import regularizer
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr, get_psnr
from utils.metric_utils import rgb_lpips, rgb_ssim
from utils.mesh_utils import get_opacity_field_from_gaussians
from utils.system_utils import load_config_from_file, merge_config
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    log_every=1000,
):
    first_iter = opt.first_iter
    args.model_path = dataset.model_path

    # Initialize models
    ## Gaussian model
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )
    glctx = dr.RasterizeGLContext()
    scene = Scene(dataset, gaussians, shuffle=True)
    ## Deform forward model
    deform = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform"
    )
    deform_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_normal",
    )
    ## Deform backward model
    deform_back = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform_back"
    )
    deform_back_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_back_normal",
    )
    ## Appearance model
    appearance = appearance_model(is_blender=dataset.is_blender)

    # Load checkpoint
    if checkpoint:
        gaussians.load_ply(checkpoint, iteration=-1)
        deform.load_weights(checkpoint, iteration=-1)
        deform_normal.load_weights(checkpoint, iteration=-1)
        deform_back.load_weights(checkpoint, iteration=-1)
        deform_back_normal.load_weights(checkpoint, iteration=-1)
        appearance.load_weights(checkpoint, iteration=-1)

    # Training setup
    gaussians.training_setup(opt)
    deform.train_setting(opt)
    deform_normal.train_setting(opt)
    deform_back.train_setting(opt)
    deform_back_normal.train_setting(opt)
    appearance.train_setting(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(
        lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000
    )
    first_iter += 1

    DPSR_ITER = opt.dpsr_iter
    ANCHOR_ITER = opt.anchor_iter
    ANCHOR_EVERY = opt.anchor_interval
    NORMAL_WARMUP_ITER = 2000

    for iteration in range(first_iter, opt.iterations + 1):
        torch.cuda.empty_cache()

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        deform.update_learning_rate(iteration)
        deform_normal.update_learning_rate(iteration)
        deform_back.update_learning_rate(iteration)
        deform_back_normal.update_learning_rate(iteration)
        appearance.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            total_frame = len(viewpoint_stack)
            time_interval = 1 / total_frame
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        fid = viewpoint_cam.fid

        # Deform the gaussians to time step t
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling, d_normal = 0.0, 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = (
                0
                if dataset.is_blender
                else torch.randn(1, 1, device="cuda").expand(N, -1)
                * time_interval
                * smooth_term(iteration)
            )
            d_xyz, d_rotation, d_scaling, _ = deform.step(
                gaussians.get_xyz.detach(), time_input + ast_noise
            )
            if iteration >= DPSR_ITER + NORMAL_WARMUP_ITER:
                d_normal = deform_normal.step(
                    gaussians.get_xyz.detach(), time_input + ast_noise
                )
            else:
                d_normal = 0.0

        # Gaussian splatting
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            dataset.is_6dof,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        losses = {}
        psnr = {}

        # Deform the time step t gaussian back to canonical space
        if iteration < opt.warm_up:
            d_xyz_back, d_rotation_back, d_scaling_back, d_normal_back = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = (
                0
                if dataset.is_blender
                else torch.randn(1, 1, device="cuda").expand(N, -1)
                * time_interval
                * smooth_term(iteration)
            )
            deformed_xyz = gaussians.get_xyz + d_xyz
            d_xyz_back, d_rotation_back, d_scaling_back, _ = deform_back.step(
                deformed_xyz.detach(), time_input + ast_noise
            )
            ## Calculate the cycle consistency loss
            cycle_loss_xyz = l1_loss(-d_xyz_back, d_xyz)
            cycle_loss_rotation = l1_loss(-d_rotation_back, d_rotation)
            cycle_loss_scaling = l1_loss(-d_scaling_back, d_scaling)
            if iteration >= DPSR_ITER + NORMAL_WARMUP_ITER:
                d_normal_back = deform_back_normal.step(
                    gaussians.get_xyz.detach(), time_input + ast_noise
                )
                cycle_loss_normal = l1_loss(-d_normal_back, d_normal)
                cycle_loss = (
                    cycle_loss_xyz
                    + cycle_loss_rotation
                    + cycle_loss_scaling
                    + cycle_loss_normal
                ) / 4.0
            else:
                cycle_loss = (
                    cycle_loss_xyz + cycle_loss_rotation + cycle_loss_scaling
                ) / 3.0
            losses["cycle_loss"] = cycle_loss

        # DPSR normal initialization, resample
        if iteration == DPSR_ITER:
            gaussians.normal_initialization(
                args, opt, dataset, deform, d_xyz, d_rotation, d_scaling
            )

        if iteration >= DPSR_ITER:
            # DPSR
            freeze_pos = iteration < DPSR_ITER + opt.normal_warm_up
            mask, mesh_image, verts, faces, _ = mesh_renderer(
                gaussians,
                d_xyz,
                d_normal,
                fid,
                glctx,
                deform_back,
                appearance,
                freeze_pos,
                dataset.white_background,
                viewpoint_cam,
            )

            ### Mask loss
            gt_mask = viewpoint_cam.gt_alpha_mask.cuda()
            mask_loss = l1_loss(mask, gt_mask)
            losses["mask_loss"] = mask_loss * 100 * opt.mask_loss_weight
            ### mesh image loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(mesh_image, gt_image)
            mesh_img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim(mesh_image, gt_image)
            )
            mesh_img_loss = mesh_img_loss * opt.mesh_img_loss_weight
            losses["mesh_img_loss"] = mesh_img_loss
            psnr["mesh_img_psnr"] = get_psnr(mesh_image.detach(), gt_image.detach())
            ## Laplacian loss
            laplacian_scale = 1000 * lp.laplacian_loss_weight
            t_iter = iteration / opt.iterations
            laplacian_loss = (
                regularizer.laplace_regularizer_const(verts, faces.long())
                * laplacian_scale
                * (1 - t_iter)
            )
            losses["laplacian_loss"] = laplacian_loss
            ## Anchoring loss
            if (
                iteration > ANCHOR_ITER
                and iteration % ANCHOR_EVERY == 0
                and lp.use_anchor > 0
            ):
                print(f"Anchoring at iteration {iteration} under fid {fid}")
                anchor_loss = gaussians.anchor_mesh(
                    verts,
                    faces,
                    deform,
                    deform_back,
                    fid,
                    search_radius=opt.anchor_search_radius,
                    topn=opt.anchor_topn,
                    bs=opt.anchor_n_1_bs,
                    increase_bs=opt.anchor_0_1_bs,
                )
                losses["anchor_loss"] = anchor_loss * 0.1

        # Gaussian image loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )
        losses["img_loss"] = img_loss

        # Log psnr
        psnr["img_psnr"] = get_psnr(image.detach(), gt_image.detach())

        ## Total loss
        loss = 0
        for k, v in losses.items():
            loss += v
        loss.backward()

        # log mesh and images
        if iteration % log_every == 0:
            with torch.no_grad():
                if iteration > DPSR_ITER:
                    mask_np = mask.repeat(1, 1, 3).detach().cpu().numpy() * 255
                    mask_np = np.clip(mask_np, 0, 255)
                    mesh_img_np = (
                        mesh_image.permute(1, 2, 0).detach().cpu().numpy() * 255
                    )
                    mesh_img_np = np.clip(mesh_img_np, 0, 255)
                    gt_mask_np = gt_mask.repeat(1, 1, 3).detach().cpu().numpy() * 255
                    imageio.imwrite(
                        osp.join(
                            args.model_path, "logs", f"mesh_image_{iteration }.png"
                        ),
                        mesh_img_np.astype(np.uint8),
                    )
                    imageio.imwrite(
                        osp.join(args.model_path, "logs", f"mask_{iteration}.png"),
                        mask_np.astype(np.uint8),
                    )
                    imageio.imwrite(
                        osp.join(args.model_path, "logs", f"gt_mask_{iteration}.png"),
                        gt_mask_np.astype(np.uint8),
                    )
            img_np = image.permute(1, 2, 0).detach().cpu().numpy() * 255
            img_np = np.clip(img_np, 0, 255)
            gt_image_np = gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255
            imageio.imwrite(
                osp.join(args.model_path, "logs", f"image_{iteration}.png"),
                img_np.astype(np.uint8),
            )
            imageio.imwrite(
                osp.join(args.model_path, "logs", f"gt_image_{iteration}.png"),
                gt_image_np.astype(np.uint8),
            )

        # Save mesh and pointcloud
        if iteration % log_every == 0:
            if iteration >= DPSR_ITER:
                verts, faces = gaussians.export_mesh(deform, deform_normal, t=fid)
                mesh = trimesh.Trimesh(
                    verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
                )
                mesh.export(
                    osp.join(args.model_path, "logs_geo", f"mesh_{iteration}.ply")
                )
            points = o3d.geometry.PointCloud()
            gaussian_pts, gaussian_nrms = gaussians.export_pointcloud(
                deform, deform_normal, t=fid
            )
            points.points = o3d.utility.Vector3dVector(
                gaussian_pts.detach().cpu().numpy()
            )
            points.normals = o3d.utility.Vector3dVector(
                gaussian_nrms.detach().cpu().numpy()
            )
            o3d.io.write_point_cloud(
                osp.join(
                    args.model_path, "logs_geo", "pointcloud_{}.ply".format(iteration)
                ),
                points,
            )

        iter_end.record()

        # Save dynamc mesh
        if iteration == opt.iterations:
            os.makedirs(os.path.join(args.model_path, "dynamic_mesh"), exist_ok=True)
            frame_num = 200
            with torch.no_grad():
                N = gaussians.get_xyz.shape[0]
                for i in range(frame_num):
                    t = i / frame_num
                    time_input = torch.ones(N, 1, device="cuda") * t
                    d_xyz, d_rotation, d_scaling, _ = deform.step(
                        gaussians.get_xyz, time_input
                    )
                    d_normal = deform_normal.step(gaussians.get_xyz, time_input)
                    t = torch.tensor([t], device="cuda")
                    verts, faces, vtx_color = mesh_renderer(
                        gaussians,
                        d_xyz,
                        d_normal,
                        t,
                        glctx,
                        deform_back,
                        appearance,
                        freeze_pos,
                        dataset.white_background,
                    )
                    vtx_color = vtx_color.detach().cpu().numpy().astype(np.uint8)
                    vertices = verts.detach().cpu().numpy()
                    faces = faces.detach().cpu().numpy()
                    ply_save_pth = os.path.join(
                        args.model_path, "dynamic_mesh", f"frame_{i}.ply"
                    )
                    save_mesh = trimesh.Trimesh(
                        vertices=vertices, faces=faces, vertex_colors=vtx_color
                    )
                    save_mesh.export(ply_save_pth)

        # Save dynamic mesh to wis3d
        if iteration == opt.iterations and dataset.save_wis3d:
            from wis3d import Wis3D

            exp_name = osp.dirname(args.model_path).split("/")[-1]
            unique_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            wis = Wis3D(
                osp.join(osp.dirname(osp.dirname(args.model_path)), "vis3d_vis"),
                f"{exp_name}-{unique_str}",
                auto_remove=False,
            )
            frame_num = 200
            with torch.no_grad():
                N = gaussians.get_xyz.shape[0]
                for i in range(frame_num):
                    t = i / frame_num
                    time_input = torch.ones(N, 1, device="cuda") * t
                    d_xyz, d_rotation, d_scaling, _ = deform.step(
                        gaussians.get_xyz, time_input
                    )
                    d_normal = deform_normal.step(gaussians.get_xyz, time_input)
                    t = torch.tensor([t], device="cuda")
                    verts, faces, vtx_color = mesh_renderer(
                        gaussians,
                        d_xyz,
                        d_normal,
                        t,
                        glctx,
                        deform_back,
                        appearance,
                        freeze_pos,
                        dataset.white_background,
                    )
                    vertices = verts.detach().cpu().numpy()
                    faces = faces.detach().cpu().numpy()
                    vtx_color = torch.clamp(vtx_color, 0.0, 1.0) * 255
                    vtx_color = vtx_color.detach().cpu().numpy().astype(np.uint8)
                    wis.add_mesh(vertices, faces, vtx_color)
                    wis.increase_scene_id()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                log_str = {}
                for k in losses.keys():
                    log_str[k] = f"{losses[k].item():.04f}"
                for k in psnr.keys():
                    log_str[k] = f"{psnr[k]:.04f}"
                progress_bar.set_postfix(log_str)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                deform_normal.save_weights(args.model_path, iteration)
                deform_back.save_weights(args.model_path, iteration)
                deform_back_normal.save_weights(args.model_path, iteration)
                appearance.save_weights(args.model_path, iteration)

            # Densification and pruning
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > opt.opacity_reset_interval else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        args.prune_threshold,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                deform.optimizer.step()
                deform_normal.optimizer.step()
                deform_back.optimizer.step()
                deform_back_normal.optimizer.step()
                appearance.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad(set_to_none=True)
                deform_normal.optimizer.zero_grad(set_to_none=True)
                deform_back.optimizer.zero_grad(set_to_none=True)
                deform_back_normal.optimizer.zero_grad(set_to_none=True)
                appearance.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_path = osp.join(args.model_path, "checkpoint")
                os.makedirs(save_path, exist_ok=True)
                gaussians.save_ply(
                    osp.join(save_path, "pointcloud_{}.ply".format(iteration))
                )

        # Finish training, testing
        with torch.no_grad():
            if iteration == opt.iterations:
                testing(
                    dataset,
                    opt,
                    pipe,
                    args,
                    scene,
                    gaussians,
                    deform,
                    deform_normal,
                    deform_back,
                    appearance,
                    background,
                )
        torch.cuda.empty_cache()


@torch.no_grad()
def testing(
    dataset,
    opt,
    pipe,
    args,
    scene,
    gaussians,
    deform,
    deform_normal,
    deform_back,
    appearance,
    background,
):
    import time

    print("Start testing...")
    viewpoint_stack = scene.getTestCameras().copy()
    glctx = dr.RasterizeGLContext()

    total_psnr, total_ssim, total_msssim, total_lpips_a, total_lpips_v = (
        [],
        [],
        [],
        [],
        [],
    )
    (
        total_psnr_mesh,
        total_ssim_mesh,
        total_msssim_mesh,
        total_lpips_a_mesh,
        total_lpips_v_mesh,
    ) = ([], [], [], [], [])
    total_times = []

    saving_path = osp.join(args.model_path, "test_results")
    os.makedirs(saving_path, exist_ok=True)
    os.makedirs(osp.join(saving_path, "dynamic_mesh"), exist_ok=True)
    os.makedirs(osp.join(saving_path, "dynamic_glb"), exist_ok=True)

    for idx, viewpoint_cam in enumerate(viewpoint_stack):
        start_time = time.time()
        fid = viewpoint_cam.fid
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        # Query the gaussian rendering
        d_xyz, d_rotation, d_scaling, _ = deform.step(
            gaussians.get_xyz.detach(), time_input
        )
        d_normal = deform_normal.step(gaussians.get_xyz.detach(), time_input)
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            dataset.is_6dof,
        )
        gs_image = render_pkg["render"]
        gs_image = torch.clamp(gs_image, 0.0, 1.0)

        # Query the mesh rendering
        _, mesh_image, verts, faces, vtx_color = mesh_renderer(
            gaussians,
            d_xyz,
            d_normal,
            fid,
            glctx,
            deform_back,
            appearance,
            False,
            dataset.white_background,
            viewpoint_cam,
        )
        total_time = time.time() - start_time
        total_times.append(total_time)

        gt_image = viewpoint_cam.original_image.cuda()
        ### Reshape
        gs_image = gs_image.permute(1, 2, 0)
        mesh_image = mesh_image.permute(1, 2, 0)
        gt_image = gt_image.permute(1, 2, 0)

        # PSNR
        psnr = get_psnr(gs_image, gt_image)
        psnr_mesh = get_psnr(mesh_image, gt_image)
        # SSIM
        ssim = rgb_ssim(gs_image.cpu(), gt_image.cpu(), 1)
        ssim_mesh = rgb_ssim(mesh_image.cpu(), gt_image.cpu(), 1)
        # MSSSIM
        ms_ssim = (
            MS_SSIM(
                gs_image.permute(2, 0, 1).unsqueeze(0).to(torch.float64),
                gt_image.permute(2, 0, 1).unsqueeze(0).to(torch.float64),
                data_range=1,
                size_average=True,
            )
            .cpu()
            .numpy()
        )
        ms_ssim_mesh = (
            MS_SSIM(
                mesh_image.permute(2, 0, 1).unsqueeze(0).to(torch.float64),
                gt_image.permute(2, 0, 1).unsqueeze(0).to(torch.float64),
                data_range=1,
                size_average=True,
            )
            .cpu()
            .numpy()
        )
        # LPIPS
        lpips_a = rgb_lpips(
            gt_image.cpu().numpy().astype(np.float32),
            gs_image.cpu().numpy().astype(np.float32),
            "alex",
            device,
        )
        lpip_a_mesh = rgb_lpips(
            gt_image.cpu().numpy().astype(np.float32),
            mesh_image.cpu().numpy().astype(np.float32),
            "alex",
            device,
        )
        lpips_v = rgb_lpips(
            gt_image.cpu().numpy().astype(np.float32),
            gs_image.cpu().numpy().astype(np.float32),
            "vgg",
            device,
        )
        lpips_v_mesh = rgb_lpips(
            gt_image.cpu().numpy().astype(np.float32),
            mesh_image.cpu().numpy().astype(np.float32),
            "vgg",
            device,
        )
        # Update
        total_psnr.append(psnr)
        total_ssim.append(ssim)
        total_msssim.append(ms_ssim)
        total_lpips_a.append(lpips_a)
        total_lpips_v.append(lpips_v)
        total_psnr_mesh.append(psnr_mesh)
        total_ssim_mesh.append(ssim_mesh)
        total_msssim_mesh.append(ms_ssim_mesh)
        total_lpips_a_mesh.append(lpip_a_mesh)
        total_lpips_v_mesh.append(lpips_v_mesh)
        # Save images
        imageio.imwrite(
            osp.join(saving_path, "gs_image_{}.png".format(idx)),
            (gs_image.cpu().numpy() * 255).astype(np.uint8),
        )
        imageio.imwrite(
            osp.join(saving_path, "mesh_image_{}.png".format(idx)),
            (mesh_image.cpu().numpy() * 255).astype(np.uint8),
        )
        imageio.imwrite(
            osp.join(saving_path, "gt_image_{}.png".format(idx)),
            (gt_image.cpu().numpy() * 255).astype(np.uint8),
        )
        print(
            f"Viewpoint {idx} PSNR {psnr:.4f} SSIM {ssim:.4f} MSSSIM {ms_ssim:.4f} LPIPS_A {lpips_a:.4f} LPIPS_V {lpips_v:.4f}"
        )
        print(
            f"Viewpoint {idx} Mesh PSNR {psnr_mesh:.4f} SSIM {ssim_mesh:.4f} MSSSIM {ms_ssim_mesh:.4f} LPIPS_A {lpip_a_mesh:.4f} LPIPS_V {lpips_v_mesh:.4f} total_time {total_time:.4f}"
        )

        # Save mesh
        vtx_color = vtx_color.detach().cpu().numpy() * 255
        vtx_color = np.clip(vtx_color, 0, 255).astype(np.uint8)
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        save_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=vtx_color
        )
        # Save in ply format
        save_mesh.export(os.path.join(saving_path, "dynamic_mesh", f"frame_{idx}.ply"))
        # Save in glb format
        save_mesh.export(osp.join(saving_path, "dynamic_glb", f"frame_{idx}.glb"))

    psnr = np.mean(total_psnr)
    ssim = np.mean(total_ssim)
    msssim = np.mean(total_msssim)
    lpips_a = np.mean(total_lpips_a)
    lpips_v = np.mean(total_lpips_v)
    psnr_mesh = np.mean(total_psnr_mesh)
    ssim_mesh = np.mean(total_ssim_mesh)
    msssim_mesh = np.mean(total_msssim_mesh)
    lpips_a_mesh = np.mean(total_lpips_a_mesh)
    lpips_v_mesh = np.mean(total_lpips_v_mesh)
    avg_time = np.mean(total_times)
    fps = len(viewpoint_stack) / np.sum(total_times)
    log_str = f"Gaussian image PSNR {psnr:.4f} SSIM {ssim:.4f} MSSSIM {msssim:.4f} LPIPS_A {lpips_a:.4f} LPIPS_V {lpips_v:.4f} \n"
    log_str += f"Mesh image PSNR {psnr_mesh:.4f} SSIM {ssim_mesh:.4f} MSSSIM {msssim_mesh:.4f} LPIPS_A {lpips_a_mesh:.4f} LPIPS_V {lpips_v_mesh:.4f} total_time {avg_time:.4f} fps {fps:.4f} \n"
    print(log_str)
    with open(osp.join(saving_path, "test_result.txt"), "w") as f:
        f.write(log_str)

    print("Testing done")


def prepare_output_and_logger(params):
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(params.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000],
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--config", type=str, default=None)

    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Load config file
    if args.config:
        config_data = load_config_from_file(args.config)
        combined_args = merge_config(config_data, args)
        args = Namespace(**combined_args)

    print("Optimizing " + args.model_path)

    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)

    # Updating save path
    unique_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_name = osp.basename(lp.source_path)
    folder_name = f"{data_name}-{unique_str}"
    if not lp.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        lp.model_path = os.path.join("./output/", unique_str[0:10])
    lp.model_path = osp.join(lp.model_path, folder_name)
    # Set up output folder
    print("Output folder: {}".format(lp.model_path))
    os.makedirs(lp.model_path, exist_ok=True)
    os.makedirs(osp.join(lp.model_path, "logs"), exist_ok=True)
    os.makedirs(osp.join(lp.model_path, "logs_geo"), exist_ok=True)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Save all parameters into file
    combined_args = vars(Namespace(**vars(lp), **vars(op), **vars(pp)))
    # Convert namespace to JSON string
    args_json = json.dumps(combined_args, indent=4)
    # Write JSON string to a text file
    with open(osp.join(lp.model_path, "cfg_args.txt"), "w") as output_file:
        output_file.write(args_json)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp,
        op,
        pp,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.log_every,
    )

    # All done
    print("\nTraining complete.")
