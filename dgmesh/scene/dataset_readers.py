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

import os, copy
import os.path as osp
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
import trimesh
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
from nvdiffrast_utils import util


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    orig_transform: Optional[np.array] = None
    alpha_mask: Optional[np.array] = None
    orig_img: Optional[np.array] = None
    K: Optional[np.array] = None
    H: Optional[int] = None
    W: Optional[int] = None
    # For fine-tuning nerf-based mesh, optional
    mesh_verts: Optional[np.array] = None
    mesh_faces: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec) 

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            ratio = 1.0
            focal_length_x = intr.params[0] * ratio
            focal_length_y = focal_length_x
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # Get origin transform
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)  # in opencv
        
        # wis.add_camera_pose(c2w) 
        c2w[:3, 1:3] *= -1 # in Blender/openGL
        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        # Mask out the background to black
        mask_path = image_path.replace('images', 'segmentations')
        mask = Image.open(mask_path)
        image_np = np.array(image)
        original_img = copy.deepcopy(image_np) / 255.0
        mask_np = np.array(mask)
        mask_np = mask_np > 0
        image_np[~mask_np] = 255  # white background
        image = Image.fromarray(image_np)
        mask_np = mask_np[..., [0]]

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid, orig_img=original_img,
                              orig_transform=c2w, alpha_mask=mask_np.astype('int'))
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", cam_scale=1.0, downsample=1.0):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            orig_cam = np.array(frame["transform_matrix"])
            
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Open-cv) (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            image = image.resize((int(image.size[0] / downsample), int(image.size[1] / downsample)), Image.Resampling.LANCZOS)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            alpha_mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                        3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, alpha_mask=alpha_mask,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1], fid=frame_time, orig_transform=orig_cam))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", downsample=1.0):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension, downsample=downsample)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension, downsample=downsample)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms_finetune(path, transformsfile, white_background, extension=".png", cam_scale=1.0, downsample=1.0, mesh_path=None):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            orig_cam = np.array(frame["transform_matrix"])
            
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            # Scale the camera position for dmetet, assume camera facing the center!!
            T = T * cam_scale

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            image = image.resize((int(image.size[0] / downsample), int(image.size[1] / downsample)), Image.Resampling.LANCZOS)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            alpha_mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                        3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy
            
            # Read mesh under current time frame
            frame_num = int(image_name.split('_')[1])
            mesh = trimesh.load(osp.join(mesh_path, f'frame_{frame_num}.ply'))
            mesh_verts = np.array(mesh.vertices)
            mesh_faces = np.array(mesh.faces)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, alpha_mask=alpha_mask,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1], fid=frame_time, orig_transform=orig_cam,
                                        mesh_verts=mesh_verts, mesh_faces=mesh_faces))

    return cam_infos


def readNerfSyntheticInfo_finetune(path, white_background, eval, extension=".png", downsample=1.0, mesh_path=None, mesh_path_test=None):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_finetune(
        path, "transforms_train.json", white_background, extension, downsample=downsample, mesh_path=mesh_path)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms_finetune(
        path, "transforms_test.json", white_background, extension, downsample=downsample, mesh_path=mesh_path_test)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path, white_background=False, downsample=1.0, nerfies_ratio=0.5):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    train_img = dataset_json['train_ids']
    val_img = dataset_json['val_ids']
    all_img = train_img + val_img
    ratio = nerfies_ratio
    train_num = len(train_img)

    if 'time_id' in meta_json[all_img[0]]:
        all_time = [meta_json[i]['time_id'] for i in all_img]
        max_time = max(all_time)
        all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    elif 'warp_id' in meta_json[all_img[0]]:
        all_time = [meta_json[i]['warp_id'] for i in all_img]
        max_time = max(all_time)
        all_time = [meta_json[i]['warp_id'] / max_time for i in all_img]

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = Image.open(image_path)
        if downsample != 1.0:
            image = image.resize((int(image.size[0] / downsample), int(image.size[1] / downsample)), Image.Resampling.LANCZOS)
        image = np.array(image)
        image_name = Path(image_path).stem

        mask_path = image_path.replace(f'/rgb/{int(1 / ratio)}x/', f'/mask-tracking/{int(1 / ratio)}x/Annotations/')
        mask = Image.open(mask_path)
        if downsample != 1.0:
            mask = mask.resize((int(mask.size[0] / downsample), int(mask.size[1] / downsample)), Image.Resampling.LANCZOS)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[..., 0] > 0  # For DEVA
        else:
            mask = mask > 0  # For SAM
        image_bg = copy.deepcopy(image) / 255.0
        if white_background:
            image[~mask] = 255.0  # white background
        else:
            image[~mask] = 0.0  # black background
            
        image = Image.fromarray((image).astype(np.uint8))
        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        principal_point = all_cam_params[idx]['principal_point']
        fid = all_time[idx]
        T = position
        R = orientation
        
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)  # in opencv
        c2w[:3, 1:3] *= -1 # in Blender/openGL

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        
        # Construct the intrinsic matrix
        K = np.array([
            [focal, 0, principal_point[0]],
            [0, focal, principal_point[1]],
            [0, 0, 1]
        ])
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, orig_img=image_bg,
                    image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                    alpha_mask = mask.astype('int').reshape(mask.shape[0], mask.shape[1], 1),
                    fid=fid, orig_transform=c2w, K=K, H=image.size[1], W=image.size[0])

        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval, white_background=False, downsample=1.0, nerfies_ratio=0.5):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path, white_background=white_background, downsample=downsample, nerfies_ratio=nerfies_ratio)  # For traditional nerfies

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Method 1
        print(f"Generating point cloud from nerfies...")
        xyz = np.load(os.path.join(path, "points.npy"))
        # TMP downsampling
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readiPhoneCameras(path, white_background=False):
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = None
    scene_center = None
    
    train_img = dataset_json['train_ids']
    val_img = dataset_json['val_ids']
    all_img = train_img + val_img
    ratio = 1.0

    train_num = len(train_img)

    all_time = [meta_json[i]['warp_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['warp_id'] / max_time for i in all_img]

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    
    for idx in range(len(all_img)):
        # Read image
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        # Mask out the background
        mask_path = image_path.replace(f'/rgb/{int(1 / ratio)}x/', f'/mask-tracking/{int(1 / ratio)}x/Annotations/')
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0] > 0  # For DEVA
        else:
            mask = mask > 0  # For SAM
        image_bg = copy.deepcopy(image) / 255.0
        if white_background:
            image[~mask] = 255.0
        else:
            image[~mask] = 0.0
        
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem
    
        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        principal_point = all_cam_params[idx]['principal_point']
        fid = all_time[idx]
        T = position
        R = orientation

        FovX = focal2fov(focal, image.size[0])
        FovY = focal2fov(focal, image.size[1])
        
        # Construct the intrinsic matrix
        K = np.array([
            [focal, 0, principal_point[0]],
            [0, focal, principal_point[1]],
            [0, 0, 1]
        ])
        
        w2c = np.eye(4)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = T
        c2w = np.linalg.inv(w2c)  # in opencv
        
        c2w[:3, 1:3] *= -1 # in Blender/openGL
        
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, orig_img=image_bg,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                              alpha_mask = mask.astype('int').reshape(mask.shape[0], mask.shape[1], 1),
                              fid=fid, orig_transform=c2w, K=K, H=image.size[1], W=image.size[0])

        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readiPhoneInfo(path, eval, white_background=False):
    print("Reading IPhone Info")
    cam_infos, train_num, scene_center, scene_scale = readiPhoneCameras(path, white_background)  # For real captured iPhone dataset

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Method 1
        print(f"Generating point cloud...")
        xyz = np.load(os.path.join(path, "points.npy"))
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNeuralActorCameras(path, transformsfile, white_background, extension=".png", downsample=1.0, load_num=1500):
    # TODO load_num to save memory
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        frames = frames[:load_num]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])
            frame_time = frame['time']
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            orig_cam = np.array(frame["transform_matrix"]) @ np.linalg.inv(util.blender2opencv.cpu().numpy())  # in OpenCV format

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            image = image.resize((int(image.size[0] / downsample), int(image.size[1] / downsample)), Image.Resampling.LANCZOS)
            im_data = np.array(image)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data
            
            # Load mask data
            if transformsfile == "transforms_train.json":
                mask_path = image_path.replace(f'/training/', f'/training_mask/Annotations/')
            else:
                mask_path = image_path.replace(f'/testing/', f'/testing_mask/Annotations/')
            alpha_mask = np.array(Image.open(mask_path))
            alpha_mask = alpha_mask[..., 0] > 0
            
            # Mask out the background
            arr[~alpha_mask] = bg
            
            alpha_mask = alpha_mask[..., None].astype('float32')
            
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            # Load intrinsics
            K = np.array(frame["intrinsic"])
            focalx = K[0, 0]
            focaly = K[1, 1]
            fovx = focal2fov(focalx, image.size[0])
            fovy = focal2fov(focaly, image.size[1])            
            FovY = fovy
            FovX = fovx
    
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, alpha_mask=alpha_mask,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1], fid=frame_time, orig_transform=orig_cam,
                                        K=K, W=image.size[0], H=image.size[1],
                                        ))

    return cam_infos


def readNeuralActorInfo(path, eval, white_background=False):
    print("Reading NeuralActor Info")
    train_cam_infos = readNeuralActorCameras(path, "transforms_train.json", white_background)
    test_cam_infos = readNeuralActorCameras(path, "transforms_test.json", white_background)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        radius = 1.
        xyz = np.random.random((num_pts, 3)) * radius * 2 - radius
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "iPhone": readiPhoneInfo,  # iPhone dataset in [https://hangg7.com/dycheck/]
    'neural_actor': readNeuralActorInfo,  # Neural Actor dataset in [https://vcai.mpi-inf.mpg.de/projects/NeuralActor/]
    'finetune-nerf': readNerfSyntheticInfo_finetune,  # Finetune a pre-trained NeRF model by only optimize the exported mesh appearance
}
