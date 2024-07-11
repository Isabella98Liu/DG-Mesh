#  created by Isabella Liu (lal005@ucsd.edu) at 2024/05/29 15:00.
#
#  Obtain mesh from Gaussians through DPSR and DiffMC and render the mesh using NVDiffRast; some other rendering functions are also included.

import io
import torch
import nvdiffrast.torch as dr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    Materials,
    TexturesVertex,
    DirectionalLights,
)

from nvdiffrast_utils import util
from scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from scene.cameras import Camera
from scene import DeformModelNormal as deform_back
from scene import AppearanceModel as appearance_net
from utils.graphics_utils import fov2focal

SMALL_NUMBER = 1e-6


def render_mask(
    glctx: dr.RasterizeGLContext,
    mesh_v_pos: torch.Tensor,
    mesh_t_pos_idx: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
    resolution: list = [800, 800],
) -> torch.Tensor:
    """Render the mask given (deformed) mesh and camera parameters

    Args:
        glctx (dr.RasterizeGLContext): Rasterization context
        mesh_v_pos (torch.tensor): Mesh vertex positions [N, 3]
        mesh_t_pos_idx (torch.tensor): Mesh triangle indices [M, 3]
        pose (torch.tensor): Camera pose in c2w format [4, 4]
        K (torch.tensor): Camera intrinsic matrix [3, 3]
        resolution (list, optional): Image resolution [H, W]

    Returns:
        torch.tensor: Mask output [H, W, 1]
    """
    # Clip space to NDC
    proj = util.K_to_projection(K, resolution[0], resolution[1])
    v_pos_clip = util.transform_pos(proj @ pose, mesh_v_pos)  # [N, 3]

    rast_out, _ = dr.rasterize(glctx, v_pos_clip, mesh_t_pos_idx, resolution=resolution)
    vtx_color = torch.ones(
        mesh_v_pos.shape, dtype=torch.float, device=v_pos_clip.device
    )
    color, _ = dr.interpolate(vtx_color[None, ...], rast_out, mesh_t_pos_idx)
    color = dr.antialias(color, rast_out, v_pos_clip, mesh_t_pos_idx)
    mask = color[0, :, :]
    mask = torch.flip(mask, dims=[0])
    return mask


def render_mesh(
    glctx: dr.RasterizeGLContext,
    mesh_v_pos: torch.Tensor,
    mesh_t_pos_idx: torch.Tensor,
    vtx_color: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
    resolution: list = [800, 800],
    whitebackground: bool = False,
) -> torch.Tensor:
    """Render the image given (deformed) mesh and camera parameters

    Args:
        glctx (dr.RasterizeGLContext): Rasterization context
        mesh_v_pos (torch.tensor): Mesh vertex positions [N, 3]
        mesh_t_pos_idx (torch.tensor): Mesh triangle indices [M, 3]
        vtx_color (torch.tensor): Vertex colors
        pose (torch.tensor): Camera pose in c2w format [4, 4]
        K (torch.tensor): Camera intrinsic matrix [3, 3]
        resolution (list, optional): Image resolution [H, W]. Defaults to [800, 800].
        whitebackground (bool, optional): Whether render white background. Defaults to False.

    Returns:
        torch.tensor: Rendering output [H, W, 3]
    """
    # Clip space to NDC
    proj = util.K_to_projection(K, resolution[0], resolution[1])
    v_pos_clip = util.transform_pos(proj @ pose, mesh_v_pos)  # [N, 3]

    rast_out, rast_deriv = dr.rasterize(
        glctx, v_pos_clip, mesh_t_pos_idx, resolution=resolution
    )
    # TODO MSAA

    # Vertex color based sampling
    output, _ = dr.interpolate(vtx_color[None, ...], rast_out, mesh_t_pos_idx)
    output = dr.antialias(output, rast_out, v_pos_clip, mesh_t_pos_idx)
    output = torch.flip(output, dims=[1])[0]

    # Mask out the background
    vtx_color = torch.ones(
        mesh_v_pos.shape, dtype=torch.float, device=v_pos_clip.device
    )
    color, _ = dr.interpolate(vtx_color[None, ...], rast_out, mesh_t_pos_idx)
    color = dr.antialias(color, rast_out, v_pos_clip, mesh_t_pos_idx)
    mask = color[0, :, :]
    mask = torch.flip(mask, dims=[0])
    if not whitebackground:
        output[~mask.bool()] = 0
    else:
        output[~mask.bool()] = 1
    output = torch.clamp(output, 0.0, 1.0)
    return output.permute(2, 0, 1)


def mesh_renderer(
    glctx: dr.RasterizeGLContext,
    gaussians: gaussian_model,
    d_xyz: torch.Tensor,
    d_normal: torch.Tensor,
    fid: torch.Tensor,
    deform_back: deform_back,
    appearance: appearance_net,
    freeze_pos: bool = False,
    whitebackground: bool = False,
    viewpoint_cam: Camera = None,
):
    """Gaussian mesh renderer

    Args:
        glctx (dr.RasterizeGLContext): Rasterization context
        gaussians (gaussian_model): Gaussians model
        d_xyz (torch.Tensor): Predicted xyz offset [N, 3]
        d_normal (torch.Tensor): Predicted normal offset [N, 3]
        fid (torch.Tensor): Time label [N, 1]
        deform_back (deform_back): Backward deformation network
        appearance (appearance_net): Appearance network
        freeze_pos (bool, optional): No xyz optimization during normal_warm_up period. Defaults to False.
        whitebackground (bool, optional): _description_. Defaults to False.
        viewpoint_cam (Camera, optional): _description_. Defaults to None.
    """
    if freeze_pos:
        dpsr_points = gaussians.get_xyz.detach() + d_xyz.detach()
    else:
        dpsr_points = gaussians.get_xyz + d_xyz
    dpsr_points = (
        dpsr_points - gaussians.gaussian_center
    ) / gaussians.gaussian_scale  # [-1, 1]
    dpsr_points = dpsr_points / 2.0 + 0.5  # [0, 1]
    dpsr_points = torch.clamp(dpsr_points, SMALL_NUMBER, 1 - SMALL_NUMBER)

    normals = gaussians.get_normal + d_normal

    # Query SDF
    psr = gaussians.dpsr(dpsr_points.unsqueeze(0), normals.unsqueeze(0))
    sign = psr[0, 0, 0, 0].detach()  # Sign for Diso is opposite to dpsr
    sign = -1 if sign < 0 else 1
    psr = psr * sign

    psr -= gaussians.density_thres_param
    psr = psr.squeeze(0)
    # Differentiable Marching Cubes
    verts, faces = gaussians.diffmc(psr, deform=None, isovalue=0.0)
    verts = verts * 2.0 - 1.0  # [-1, 1]
    verts = verts * gaussians.gaussian_scale + gaussians.gaussian_center
    verts = verts.to(torch.float32)
    faces = faces.to(torch.int32)
    # Deform mesh vertex back to canonical mesh and query vertex color
    N = verts.shape[0]
    time_input = fid.unsqueeze(0).expand(N, -1)
    mesh_deform_back_dxyz, _, _, _ = deform_back.step(verts.detach(), time_input)
    mesh_canonical_xyz = verts + mesh_deform_back_dxyz
    vtx_color = appearance.step(mesh_canonical_xyz, time_input)
    if viewpoint_cam is not None:
        # Compose projection matrix
        if viewpoint_cam.K is not None:
            K = torch.tensor(viewpoint_cam.K).float().to("cuda")
        else:
            fovx = viewpoint_cam.FoVx
            fovy = viewpoint_cam.FoVy
            focalx = fov2focal(fovx, viewpoint_cam.image_width)
            focaly = fov2focal(fovy, viewpoint_cam.image_height)
            K = (
                torch.tensor(
                    [
                        [focalx, 0, viewpoint_cam.image_width / 2],
                        [0, focaly, viewpoint_cam.image_height / 2],
                        [0, 0, 1],
                    ]
                )
                .float()
                .to("cuda")
            )

        c2w_blender = (
            torch.tensor(viewpoint_cam.orig_transform).cuda().float()
        )  # blender/OpenGL camera
        c2w_opencv = c2w_blender @ util.blender2opencv
        w2c_blender = util.opencv2blender @ torch.inverse(c2w_opencv)
        pose = w2c_blender
        # Render mask
        mask = render_mask(
            glctx,
            verts,
            faces,
            pose,
            K,
            resolution=[viewpoint_cam.image_height, viewpoint_cam.image_width],
        )
        mask = mask[..., [0]]
        # Render image
        # Render
        mesh_image = render_mesh(
            glctx,
            verts,
            faces,
            vtx_color,
            pose,
            K,
            resolution=[viewpoint_cam.image_height, viewpoint_cam.image_width],
            whitebackground=whitebackground,
        )
        return mask, mesh_image, verts, faces, vtx_color
    else:
        return verts, faces, vtx_color


def mesh_shape_renderer(
    vertices: torch.Tensor, faces: torch.Tensor, viewpoint_cam: Camera
) -> torch.Tensor:
    """Render mesh shape using PyTorch3D

    Args:
        vertices (torch.Tensor): Mesh vertices [N, 3]
        faces (torch.Tensor): Mesh faces [M, 3]
        viewpoint_cam (Camera): Viewpoint camera

    Returns:
        torch.Tensor: Mesh shape image [H, W, 3]
    """
    colors = torch.ones(1, vertices.shape[0], 3).to("cuda")
    textures = TexturesVertex(verts_features=colors)
    mesh = Meshes(
        verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0), textures=textures
    )

    c2w_blender = torch.tensor(viewpoint_cam.orig_transform).cuda().float()
    c2w_p3d = c2w_blender @ util.blender2p3d
    w2c_p3d = torch.inverse(c2w_p3d)
    R = w2c_p3d[:3, :3].T.clone().detach().unsqueeze(0)
    T = w2c_p3d[:3, 3].clone().detach().unsqueeze(0)

    if viewpoint_cam.K is None:
        fovx = viewpoint_cam.FoVx
        fovy = viewpoint_cam.FoVy
        fx = fov2focal(fovx, viewpoint_cam.image_width)
        fy = fov2focal(fovy, viewpoint_cam.image_height)
        cx = viewpoint_cam.image_width / 2
        cy = viewpoint_cam.image_height / 2
    else:
        K = torch.tensor(viewpoint_cam.K).float().to("cuda")
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
    cameras = PerspectiveCameras(
        R=R,
        T=T,
        focal_length=((fx, fy),),
        principal_point=((viewpoint_cam.image_height - cx, cy),),
        image_size=((viewpoint_cam.image_height, viewpoint_cam.image_width),),
        device=torch.device("cuda"),
        in_ndc=False,
    )

    light_pos = (
        torch.tensor(viewpoint_cam.orig_transform[:3, 3])
        .float()
        .to("cuda")
        .unsqueeze(0)
    )
    # Get mesh vertices rought center
    mesh_center = vertices.mean(0)
    light_dir = mesh_center - light_pos
    lights = DirectionalLights(
        device=torch.device("cuda"),
        direction=-light_dir,
    )

    materials = Materials(
        device="cuda",
        specular_color=[[0.2, 0.2, 0.2]],
        shininess=10,  # Control the shininess for the specular component
    )
    raster_settings = RasterizationSettings(
        image_size=(viewpoint_cam.image_height, viewpoint_cam.image_width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=torch.device("cuda"),
            cameras=cameras,
            lights=lights,
            materials=materials,
        ),
    )
    mesh_img = renderer(mesh)[0][..., :3].flip(1)
    return mesh_img


def pointcloud_renderer(points: torch.Tensor, viewpoint_cam: Camera) -> np.ndarray:
    """Render the Gaussians point cloud image

    Args:
        points (torch.Tensor): Gaussian points positions [N, 3]
        viewpoint_cam (Camera): Viewpoint camera

    Returns:
        np.ndarray: Point cloud image [H, W, 3]
    """
    points = points.detach().cpu().numpy()

    # Obtain camera extrinsics
    c2w_blender = torch.tensor(viewpoint_cam.orig_transform).cuda().float()
    c2w_opencv = c2w_blender @ util.blender2opencv
    w2c_opencv = torch.inverse(c2w_opencv)
    R = w2c_opencv[:3, :3].detach().cpu().numpy()
    T = w2c_opencv[:3, 3].detach().cpu().numpy().reshape(-1, 1)

    fovx = viewpoint_cam.FoVx
    fovy = viewpoint_cam.FoVy
    fx = fov2focal(fovx, viewpoint_cam.image_width)
    fy = fov2focal(fovy, viewpoint_cam.image_height)
    cx = viewpoint_cam.image_width / 2
    cy = viewpoint_cam.image_height / 2

    # Obtain camera intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Project points to 2D
    points_h = np.hstack((points, np.ones((points.shape[0], 1)))).T
    projection_matrix = K @ np.hstack((R, T))
    projected_points_h = projection_matrix @ points_h
    projected_points = projected_points_h[:2] / projected_points_h[2]

    fig = plt.figure(figsize=(8, 8))
    plt.xlim(0, 800)
    plt.ylim(0, 800)
    plt.autoscale(False)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().invert_yaxis()  # Invert y axis to match image coordinates
    plt.scatter(projected_points[0], projected_points[1], s=0.06, c="blue")
    # plt.gca().invert_yaxis()  # Invert y axis to match image coordinates
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pointcloud_img = np.array(Image.open(buf))[..., :3]
    # Close the buffer
    buf.close()
    plt.close(fig)

    return pointcloud_img
