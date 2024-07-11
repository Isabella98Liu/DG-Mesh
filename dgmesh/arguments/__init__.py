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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._expname = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.data_type = ""
        self.data_mask = False
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.is_6dof = False
        self.downsample = 1.0
        self.nerfies_ratio = 0.5
        
        self.prune_threshold = 0.005
        
        self.laplacian_loss_weight = 1.0
        self.use_anchor = 1.0
        
        # For DPSR
        self.grid_res = 256
        self.gaussian_ratio = 1.5
        self.gaussian_center = [0.0, 0.0, 0.0]
        
        # Other 
        self.save_wis3d = False

        # Only for pre-train fine-tuning
        self.pretrain_mesh_path = ""
        self.pretrain_mesh_path_test = ""
        self.pretrained_type = "dgmesh"
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.first_iter = -1
        
        self.iterations = 40_000
        self.warm_up = 3_000
        self.normal_warm_up = 1_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 40_000
        
        self.apperance_lr_init = 0.00016
        self.apperance_lr_final = 0.0000016
        self.apperance_lr_delay_mult = 0.01
        self.apperance_lr_max_steps = 40_000
        
        self.deform_lr_max_steps = 40_000
        
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.anchor_iter = 10000
        self.densify_grad_threshold = 0.0002
        
        # For anchoring
        self.anchor_search_radius = 0.0005
        self.anchor_topn = 2
        self.anchor_n_1_bs = 512
        self.anchor_0_1_bs = 1024
        
        # For dpsr
        self.dpsr_iter = 5000
        self.anchor_iter = 8000
        self.init_density_threshold = 0.05
        self.dpsr_sig = 0.5
        
        # Different loss weight
        self.mask_loss_weight = 10.0
        self.mesh_img_loss_weight = 1.0
        
        # Compute gaussain scale and center
        self.anchor_interval = 100
        
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
