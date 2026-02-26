"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import argparse

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# ---------------- DemoSpeedUp / Speedup args (optional) ----------------
# These flags are OPTIONAL. They are here so you can run:
#   (1) original DP:   python train.py --config-name=robot_dp_14 ...
#   (2) nash DP:       python train.py --config-name=robot_dp_nash ...
#   (3) speedup DP:    python train.py --config-name=robot_dp_speedup ...
#       label stage:   python train.py --config-name=robot_dp_speedup --label --label-ckpt-path xxx.ckpt
# Note: we strip these flags before Hydra parses argv.
def _parse_speedup_flags(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--label', action='store_true', help='Label entropy for DemoSpeedUp and exit')
    parser.add_argument('--speedup', action='store_true', help='Enable speedup training (override cfg.speedup.enable)')
    parser.add_argument('--label-ckpt-path', type=str, default=None, help='Checkpoint used for labeling')
    parser.add_argument('--num-entropy-samples', type=int, default=None)
    parser.add_argument('--entropy-quantile', type=float, default=None)
    parser.add_argument('--low-v', type=int, default=None)
    parser.add_argument('--high-v', type=int, default=None)
    parser.add_argument('--apply-to-val', action='store_true')
    args, unknown = parser.parse_known_args(argv)
    return args, unknown

_SPEEDUP_ARGS, _HYDRA_ARGV = _parse_speedup_flags(sys.argv[1:])
sys.argv = [sys.argv[0]] + _HYDRA_ARGV



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = getattr(cfg, 'head_camera_type', None)
    head_camera_cfg = None
    if head_camera_type is not None:
        head_camera_cfg = get_camera_config(head_camera_type)
        cfg.task.image_shape = [3, head_camera_cfg['h'], head_camera_cfg['w']]
        try:
            cfg.task.shape_meta.obs.head_cam.shape = [3, head_camera_cfg['h'], head_camera_cfg['w']]
        except Exception:
            # some tasks may not have head_cam in shape_meta
            pass
    OmegaConf.resolve(cfg)

    # re-apply camera-derived shapes after resolve (in case resolvers overwrote them)
    if head_camera_cfg is not None:
        cfg.task.image_shape = [3, head_camera_cfg['h'], head_camera_cfg['w']]
        try:
            cfg.task.shape_meta.obs.head_cam.shape = [3, head_camera_cfg['h'], head_camera_cfg['w']]
        except Exception:
            pass

    # Optional CLI overrides (keep config-based switching as the default)
    OmegaConf.set_struct(cfg, False)
    if getattr(_SPEEDUP_ARGS, 'label', False) or getattr(_SPEEDUP_ARGS, 'speedup', False):
        if getattr(cfg, 'speedup', None) is None:
            cfg.speedup = {}
        if _SPEEDUP_ARGS.label:
            cfg.speedup.label = True
        if _SPEEDUP_ARGS.speedup:
            cfg.speedup.enable = True
        if _SPEEDUP_ARGS.label_ckpt_path is not None:
            cfg.speedup.label_ckpt_path = _SPEEDUP_ARGS.label_ckpt_path
        if _SPEEDUP_ARGS.num_entropy_samples is not None:
            cfg.speedup.num_entropy_samples = int(_SPEEDUP_ARGS.num_entropy_samples)
        if _SPEEDUP_ARGS.entropy_quantile is not None:
            cfg.speedup.entropy_quantile = float(_SPEEDUP_ARGS.entropy_quantile)
        if _SPEEDUP_ARGS.low_v is not None:
            cfg.speedup.low_v = int(_SPEEDUP_ARGS.low_v)
        if _SPEEDUP_ARGS.high_v is not None:
            cfg.speedup.high_v = int(_SPEEDUP_ARGS.high_v)
        if _SPEEDUP_ARGS.apply_to_val:
            cfg.speedup.apply_to_val = True

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()