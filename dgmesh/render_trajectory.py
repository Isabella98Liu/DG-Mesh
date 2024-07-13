#  created by Isabella Liu (lal005@ucsd.edu) at 2024/05/29 17:10.
#
#  Rendering the DG-Mesh in a trajectory


import os, sys
import copy
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
import nvdiffrast.torch as dr

from scene import Scene
from scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from scene import DeformModelNormal as deform_model
from scene import DeformModelNormalSep as deform_model_sep
from scene import AppearanceModel as appearance_model
from utils.renderer import mesh_renderer, mesh_shape_renderer, pointcloud_renderer
from utils.general_utils import safe_state
from utils.system_utils import load_config_from_file, merge_config
from utils.camera_utils import get_camera_trajectory_pose
from arguments import ModelParams, PipelineParams, OptimizationParams


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def rendering_trajectory(
    dataset,
    opt,
    pipe,
    checkpoint,
    camera_radius,
    camera_elevation,
    camera_lookat,
    total_frames,
    fps=24,
):
    args.model_path = dataset.model_path

    # Load models
    ## Gaussian model
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )
    glctx = dr.RasterizeGLContext()
    scene = Scene(dataset, gaussians, shuffle=False)
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
    ## Load checkpoint
    if checkpoint:
        gaussians.load_ply(checkpoint, iteration=-1)
        deform.load_weights(checkpoint, iteration=-1)
        deform_normal.load_weights(checkpoint, iteration=-1)
        deform_back.load_weights(checkpoint, iteration=-1)
        deform_back_normal.load_weights(checkpoint, iteration=-1)
        appearance.load_weights(checkpoint, iteration=-1)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Compose camera trajectory
    camera_poses = get_camera_trajectory_pose(
        camera_radius, camera_elevation, total_frames, look_at=camera_lookat
    )
    viewpoint_cam = scene.getTestCameras()[
        0
    ]  # Use the intrinsics from the first camera in the test cameras

    # Create folders
    image_folder = osp.join(dataset.model_path, "images")
    os.makedirs(image_folder, exist_ok=True)
    final_images = []

    for idx, pose in tqdm(enumerate(camera_poses)):
        render_cam = copy.deepcopy(viewpoint_cam)
        render_cam.orig_transform = pose

        fid = torch.tensor([idx / total_frames], device="cuda")
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        # Query the gaussians
        d_xyz, d_rotation, d_scaling, _ = deform.step(
            gaussians.get_xyz.detach(), time_input
        )
        d_normal = deform_normal.step(gaussians.get_xyz.detach(), time_input)

        # Query the mesh rendering rgb
        mask, mesh_image, verts, faces, vtx_color = mesh_renderer(
            glctx,
            gaussians,
            d_xyz,
            d_normal,
            fid,
            deform_back,
            appearance,
            False,
            True,
            render_cam,
        )
        mesh_image_np = mesh_image.permute(1, 2, 0).detach().cpu().numpy() * 255

        # Render the mesh itself
        mesh_image_shape = mesh_shape_renderer(
            verts, faces, render_cam, mask=mask, orig_img=None, bg_color=bg_color
        )
        mesh_image_shape_np = mesh_image_shape.detach().cpu().numpy() * 255

        # Obtain the gaussian point cloud image
        gaussian_point_img = pointcloud_renderer(gaussians.get_xyz + d_xyz, render_cam)
        gaussian_point_img_np = gaussian_point_img

        # Compose the final image
        final_img = np.hstack(
            [mesh_image_np, mesh_image_shape_np, gaussian_point_img_np]
        )
        img_save_path = osp.join(image_folder, f"{idx:04d}.png")
        imageio.imwrite(img_save_path, final_img.astype(np.uint8))

        final_images.append(final_img)

    # Save the final video
    final_images = np.stack(final_images).astype(np.uint8)

    # Save the gif
    with imageio.get_writer(
        osp.join(dataset.model_path, "video.gif"), fps=fps, codec="libx264", loop=0
    ) as writer:
        for img in final_images:
            writer.append_data(img)

    # Save the mp4
    with imageio.get_writer(
        osp.join(dataset.model_path, "video.mp4"), fps=fps, codec="libx264"
    ) as writer:
        for img in final_images:
            writer.append_data(img)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--camera_radius", type=float, default=4.0)
    parser.add_argument("--camera_lookat", type=float, nargs="+", default=[0, 0, 0])
    parser.add_argument("--camera_elevation", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total_frames", type=int, default=100)

    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args(sys.argv[1:])

    # Load config file
    if args.config:
        config_data = load_config_from_file(args.config)
        combined_args = merge_config(config_data, args)
        args = Namespace(**combined_args)

    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)

    # Updating save path
    unique_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_name = osp.basename(lp.source_path)
    folder_name = f"rendering-traj-{data_name}-{unique_str}"
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
    rendering_trajectory(
        lp,
        op,
        pp,
        args.start_checkpoint,
        args.camera_radius,
        args.camera_elevation,
        args.camera_lookat,
        args.total_frames,
        args.fps,
    )

    # All done
    print("\nRendering complete.")
