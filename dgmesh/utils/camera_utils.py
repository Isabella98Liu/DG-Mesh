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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
from nvdiffrast_utils.util import opencv2blender
import json
import torch

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # if resized_image_rgb.shape[1] == 4:
        # loaded_mask = resized_image_rgb[3:4, ...]
    if cam_info.alpha_mask is not None:
        # loaded_mask = torch.tensor(cam_info.alpha_mask).squeeze(0).permute(2, 0, 1)  # [H, W, 1]
        # loaded_mask = torch.tensor(cam_info.alpha_mask).unsqueeze(0).permute(1, 2, 0)  # [H, W, 1]  # real
        loaded_mask = torch.tensor(cam_info.alpha_mask)  # [H, W, 1]  # sim

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, orig_transform=cam_info.orig_transform, orig_img=cam_info.orig_img,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  depth=cam_info.depth, K=cam_info.K, W=cam_info.W, H=cam_info.H, mesh_verts=cam_info.mesh_verts, mesh_faces=cam_info.mesh_faces)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )


def compute_pose_matrix(camera_pos, look_at, up):
    camera_pos = np.array(camera_pos)
    look_at = np.array(look_at)
    up = np.array(up)
    
    look_at = look_at - camera_pos
    look_at /= np.linalg.norm(look_at)
    up = np.array([0, 0, 1])
    right = np.cross(look_at, up)
    right /= np.linalg.norm(right)
    up_new = np.cross(right, look_at)
    up_new /= np.linalg.norm(up_new)
    cam_rot = np.array([right, up_new, -look_at]).transpose()
    
    camera_trans = np.vstack([np.hstack([cam_rot, camera_pos.reshape(-1, 1)]), np.array([0, 0, 0, 1])])
    return camera_trans


def get_camera_trajectory_pose(radius, elevation, total_frames, look_at=[0, 0, 0], up_vector=[0, 0, 1]):
    # Generate poses for camera trajectory
    camera_poses = []
    for i in range(total_frames):
        theta = 2 * np.pi * i / total_frames
        r = np.sqrt(radius ** 2 - elevation ** 2)
        camera_loc = [r * np.sin(theta), -r * np.cos(theta), elevation]
        camera_pose = compute_pose_matrix(camera_loc, look_at, up_vector)
        camera_poses.append(camera_pose)
    return camera_poses