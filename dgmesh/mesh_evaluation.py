#  created by Isabella Liu (lal005@ucsd.edu) at 2024/03/08 14:38.
#
#  Evaluate Chamfer distance between GT mesh and predicted mesh.


import numpy as np
import glob
from chamferdist import ChamferDistance
import torch
from wis3d import Wis3D
import trimesh
from tqdm import tqdm
import configargparse
import os
import time
import os.path as osp
from utils.emd_utils import emdModule
import json
from nvdiffrast_utils.util import blender2opencv
from metrics.evaluation_metrics import emd_cd
from utils.pose_utils import (
    rotate_mtx_dgmesh,
    rotate_mtx_hexplane,
    rotate_mtx_tineuvox,
    rotate_mtx_dnerf,
    rotate_mtx_kplane,
    rotate_mtx_deformable_gaussian,
)


def eval_distance(
    chamferDist,
    emd,
    gt_mesh,
    eval_mesh,
    convert,
    rotate_mtx,
    cam_origin=None,
    emd_sample=8192,
    wis3d=None,
):
    # Read GT mesh obj file
    gt_mesh = trimesh.load(gt_mesh, force="mesh")
    gt_mesh_pts = torch.tensor(gt_mesh.vertices).cuda().float()
    # Convert blender camera origin to opencv camera origin
    if cam_origin is not None:
        cam_origin = np.hstack((cam_origin, 1))
        cam_origin = blender2opencv @ torch.tensor(cam_origin).cuda().float()
        cam_origin = cam_origin[:3]
        cam_origin = torch.inverse(rotate_mtx_dgmesh) @ cam_origin
        gt_mesh_pts = gt_mesh_pts - torch.tensor(cam_origin).cuda().float()

    # Read eval mesh ply file
    eval_mesh = trimesh.load(eval_mesh, force="mesh")
    eval_mesh_pts = torch.tensor(eval_mesh.vertices).cuda().float()

    if convert:
        eval_mesh_pts = rotate_mtx @ eval_mesh_pts.T
        eval_mesh_pts = eval_mesh_pts.T

    if wis3d is not None:
        wis3d.add_point_cloud(gt_mesh_pts.cpu().numpy(), name="gt_mesh")
        wis3d.add_point_cloud(eval_mesh_pts.cpu().numpy(), name="eval_mesh")
        wis3d.increase_scene_id()

    # Evaluate Chamfer distance
    chamfer_dist = (
        chamferDist(gt_mesh_pts[None], eval_mesh_pts[None], point_reduction="mean")
        + chamferDist(eval_mesh_pts[None], gt_mesh_pts[None], point_reduction="mean")
    ) / 2

    # Evaluate EMD distance
    gt_mesh_sampled_pts = torch.tensor(gt_mesh.sample(emd_sample)).cuda().float()
    if cam_origin is not None:
        gt_mesh_sampled_pts = (
            gt_mesh_sampled_pts - torch.tensor(cam_origin).cuda().float()
        )
    eval_mesh_sampled_pts = torch.tensor(eval_mesh.sample(emd_sample)).cuda().float()
    if convert:
        eval_mesh_sampled_pts = rotate_mtx @ eval_mesh_sampled_pts.T
        eval_mesh_sampled_pts = eval_mesh_sampled_pts.T.contiguous()

    if wis3d is not None:
        wis3d.add_point_cloud(gt_mesh_sampled_pts.cpu().numpy(), name="gt_mesh_sampled")
        wis3d.add_point_cloud(
            eval_mesh_sampled_pts.cpu().numpy(), name="eval_mesh_sampled"
        )
        wis3d.increase_scene_id()

    emd_dict = emd_cd(
        gt_mesh_sampled_pts[None], eval_mesh_sampled_pts[None], 128, accelerated_cd=True
    )
    emd_dist = emd_dict["EMD"].item()

    return chamfer_dist.item(), emd_dist


def evaluation(
    gt_mesh_path, eval_mesh_path, eval_model_type, wis3d=None, fix_mesh=False
):
    # Decide whether convert coordinate
    assert eval_model_type in [
        "dgmesh",
        "hexplane",
        "tineuvox",
        "dnerf",
        "kplane",
        "deformable_gaussian",
    ], "eval_model_type not supported!"
    if eval_model_type == "dgmesh":
        convert = True
        rotate_mtx = rotate_mtx_dgmesh
    elif eval_model_type == "hexplane":
        convert = True
        rotate_mtx = rotate_mtx_hexplane
    elif eval_model_type == "tineuvox":
        convert = True
        rotate_mtx = rotate_mtx_tineuvox
    elif eval_model_type == "dnerf":
        convert = True
        rotate_mtx = rotate_mtx_dnerf
    elif eval_model_type == "kplane":
        convert = True
        rotate_mtx = rotate_mtx_kplane
    elif eval_model_type == "deformable_gaussian":
        convert = True
        rotate_mtx = rotate_mtx_deformable_gaussian

    # Load mesh list
    gt_meshes_list = sorted(glob.glob(gt_mesh_path + "/*.obj"))
    eval_mesh_list = sorted(glob.glob(eval_mesh_path + "/*.ply"))
    assert len(gt_meshes_list) == len(
        eval_mesh_list
    ), "Number of GT meshes and predicted meshes not equal!"

    # Load camera origin (some dataset will have camera origin not align with GT mesh origin)
    json_path = osp.join(osp.dirname(gt_mesh_path), "transforms_train.json")
    json_content = json.load(open(json_path, "r"))
    try:
        cam_origin = np.array(json_content["camera_origin"])
    except:
        cam_origin = None

    chamferDist = ChamferDistance()
    emd = emdModule()
    chamfer_dist_list = []
    emd_list = []

    print("Start evaluation...")

    for idx, (gt_mesh, eval_mesh) in tqdm(
        enumerate(zip(gt_meshes_list, eval_mesh_list))
    ):

        # Calculate Chamfer distance
        chamfer_dist, emd = eval_distance(
            chamferDist,
            emd,
            gt_mesh,
            eval_mesh,
            convert,
            rotate_mtx,
            cam_origin=cam_origin,
            wis3d=wis3d,
        )
        chamfer_dist_list += [chamfer_dist]
        emd_list += [emd]

        # Print out evaluation results currently
        print(f"Item {idx}: CD {chamfer_dist:.10f}, EMD {emd:.4f}")

    avg_chamfer_dist = np.mean(chamfer_dist_list)
    avg_emd_dist = np.mean(emd_list)

    print(f"Average Chamfer distance: {avg_chamfer_dist:.4f}")
    print(f"Average EMD: {avg_emd_dist:.4f}")

    return avg_chamfer_dist, chamfer_dist_list, avg_emd_dist, emd_list


def main():

    parser = configargparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, help="Path to predicted mesh folder", required=True
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        help="Type of evaluation, [dgmesh, hexplane, tineuvox, dnerf, kplane, deformable_gaussian]",
        required=True,
    )
    parser.add_argument(
        "--debug", type=bool, help="Visualize mesh comparison in wis3d", default=False
    )
    parser.add_argument(
        "--fix_mesh", type=bool, help="Fix non water-tight mesh", default=False
    )

    args = parser.parse_args()

    # Search the GT path
    args.gt_path = osp.join(args.path, "gt")
    if args.eval_type == "dgmesh":
        args.pred_path = osp.join(args.path, "DGMesh")
    elif args.eval_type == "hexplane":
        args.pred_path = osp.join(args.path, "HexPlane")
    elif args.eval_type == "tineuvox":
        args.pred_path = osp.join(args.path, "TiNeuVox")
    elif args.eval_type == "dnerf":
        args.pred_path = osp.join(args.path, "D-NeRF")
    elif args.eval_type == "kplane":
        args.pred_path = osp.join(args.path, "K-Plane")

    assert osp.exists(args.pred_path), "Predicted results path not found!"

    args.log_folder = osp.join(args.pred_path, "results")
    args.pred_path = osp.join(args.pred_path, "dynamic_mesh")

    print(
        f"GT path: {args.gt_path} \nPred path: {args.pred_path} \nLog folder: {args.log_folder}"
    )

    # Create log folder and wis3d debug folder
    item_name = args.gt_path.split("/")[-2]
    log_folder = osp.join(
        args.log_folder,
        item_name + time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime()),
    )
    os.makedirs(log_folder, exist_ok=True)
    wis3d = (
        Wis3D(osp.join(osp.dirname(log_folder), "wis3d"), "debug")
        if args.debug
        else None
    )

    # Run evaluation
    avg_chamfer_dist, _, avg_emd, _ = evaluation(
        args.gt_path, args.pred_path, args.eval_type, wis3d, args.fix_mesh
    )

    # Write evaluation results to log file
    with open(osp.join(log_folder, "eval_results.txt"), "w") as f:
        f.write(f"GT source: {args.gt_path}\n")
        f.write(f"Pred source: {args.pred_path}\n")
        f.write(f"Average Chamfer distance: {avg_chamfer_dist:.10f}\n")
        f.write(f"Average EMD: {avg_emd:.4f}\n")


if __name__ == "__main__":
    main()
