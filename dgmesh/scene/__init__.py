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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_dpsr_dynamic_anchor import GaussianModelDPSRDynamicAnchor
from scene.deform_model import DeformModel, DeformModelNormal, DeformModelNormalSep
from scene.appearance_model import AppearanceModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.data_type == "iPhone":
            print("Assuming iPhone data set!")
            scene_info = sceneLoadTypeCallbacks["iPhone"](args.source_path, args.eval, args.white_background)

        elif args.data_type == "NeuralActor":
            print("Assuming Neural Actor dataset!")
            scene_info = sceneLoadTypeCallbacks["neural_actor"](args.source_path, args.eval, args.white_background)
        
        elif args.data_type == "finetune-nerf":
            print("Assuming fine-tuning pre-trained nerf mesh!")
            scene_info = sceneLoadTypeCallbacks["finetune-nerf"](args.source_path, args.white_background, args.eval, downsample=args.downsample, mesh_path=args.pretrain_mesh_path, mesh_path_test=args.pretrain_mesh_path_test)
        
        elif args.data_type == "Nerfies":
            assert os.path.exists(os.path.join(args.source_path, "dataset.json")), "Nerfies data set should have a dataset.json file!"
            print("Assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval, args.white_background, args.downsample, args.nerfies_ratio)

        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            print("Assuming Colmap dataset!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, downsample=args.downsample)
            
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
                        
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
            
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
            
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                                args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        if self.gaussians is not None:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                    "point_cloud.ply"),
                                        og_number_points=len(scene_info.point_cloud.points))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        self.scene_info = scene_info

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_dynamic(self, iteration, deform, time=0.0):
        point_cloud_path = os.path.join(self.model_path, "point_cloud_dynamic/iteration_{}".format(iteration))
        self.gaussians.save_ply_dynamic(os.path.join(point_cloud_path, "point_cloud.ply"), deform=deform, time=time)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
