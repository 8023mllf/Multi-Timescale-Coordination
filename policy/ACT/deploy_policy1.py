import sys
import numpy as np
import torch
import os
import pickle
import cv2
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACT
import copy
from argparse import Namespace


def encode_obs(observation):
    """
    将 RoboTwin 环境返回的 observation 编码成 ACT 需要的输入格式。

    约定：
    - 头部相机 head_camera 一定存在，作为主视角；
    - 左/右腕部相机 left_camera / right_camera 是可选的：
      * 如果存在（相机在仿真配置中开启），则使用腕部图像；
      * 如果不存在（相机在仿真配置中关闭），则直接使用头部图像作为占位，
        不会去访问 observation["observation"]["left_camera"] 之类的 key。
    """
    obs_dict = observation.get("observation", {})

    # === 必须存在的头部相机 ===
    if "head_camera" not in obs_dict or "rgb" not in obs_dict["head_camera"]:
        raise KeyError(
            "Observation does not contain required 'head_camera.rgb' field. "
            f"Available keys: {list(obs_dict.keys())}"
        )

    head_rgb = obs_dict["head_camera"]["rgb"]
    # 统一 resize 到 (640, 480)
    head_cam = cv2.resize(
        head_rgb,
        (640, 480),
        interpolation=cv2.INTER_LINEAR
    )
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0  # (C, H, W) in [0, 1]

    # === 可选的腕部相机：如果不存在就用头部图像占位 ===
    def get_cam_rgb(cam_key: str):
        cam = obs_dict.get(cam_key, None)
        if cam is None or "rgb" not in cam:
            # 仿真配置关闭该相机时，对应 key 不存在，直接用头部图像代替
            return head_rgb
        return cam["rgb"]

    left_rgb = get_cam_rgb("left_camera")
    right_rgb = get_cam_rgb("right_camera")

    left_cam = cv2.resize(
        left_rgb,
        (640, 480),
        interpolation=cv2.INTER_LINEAR
    )
    right_cam = cv2.resize(
        right_rgb,
        (640, 480),
        interpolation=cv2.INTER_LINEAR
    )

    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0

    # === 关节状态（保持原来的拼接方式） ===
    qpos = (
        observation["joint_action"]["left_arm"]
        + [observation["joint_action"]["left_gripper"]]
        + observation["joint_action"]["right_arm"]
        + [observation["joint_action"]["right_gripper"]]
    )

    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": qpos,
    }


def get_model(usr_args):
    return ACT(usr_args, Namespace(**usr_args))


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    # instruction = TASK_ENV.get_instruction()

    # Get action from model
    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros(
            [
                model.max_timesteps,
                model.max_timesteps + model.num_queries,
                model.state_dim,
            ]
        ).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0
