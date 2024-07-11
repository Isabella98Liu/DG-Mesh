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

from errno import EEXIST
from os import makedirs, path
import os
import yaml


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def load_config_from_file(config_file):
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        print(f"Configuration loaded from {config_file}")
    else:
        config_data = {}
        print(f"Configuration file {config_file} not found. Using default settings.")
    return config_data


def merge_config(file_config, cli_args):
    combined_args = {}
    for k, v in vars(cli_args).items():
        if k in file_config.keys():
            combined_args[k] = file_config[k]
        else:
            combined_args[k] = v
    return combined_args