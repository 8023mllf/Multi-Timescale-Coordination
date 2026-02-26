import numpy as np
import torch
import cv2
from argparse import Namespace

from .act_policy import ACT
from .dp_policy import DP


def encode_obs(observation):
    def _prep_cam(cam_key: str):
        rgb = observation["observation"][cam_key]["rgb"]
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
        rgb = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0  # (3,H,W)
        return rgb

    head_cam = _prep_cam("head_camera")
    left_cam = _prep_cam("left_camera")
    right_cam = _prep_cam("right_camera")
    front_cam = _prep_cam("front_camera")

    # 用更可靠的 vector（你打印出来 shape=(14,)）
    qpos = np.array(observation["joint_action"]["vector"], dtype=np.float32)

    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "front_cam": front_cam,   # DP 默认用这个
        "qpos": qpos,
    }


def get_model(usr_args):
    policy_class = str(usr_args.get("policy_class", "ACT")).upper()
    if policy_class == "DP":
        return DP(usr_args, Namespace(**usr_args))
    return ACT(usr_args, Namespace(**usr_args))


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)

    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    # 如果模型实现了 reset()，优先使用
    if hasattr(model, "reset") and callable(getattr(model, "reset")):
        model.reset()
        return

    # 兼容旧 ACT 行为
    if getattr(model, "temporal_agg", False):
        model.all_time_actions = torch.zeros([
            model.max_timesteps,
            model.max_timesteps + model.num_queries,
            model.state_dim,
        ]).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0
