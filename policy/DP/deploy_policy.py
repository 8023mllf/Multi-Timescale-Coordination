import numpy as np
from .dp_model import DP
import yaml
import os

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    # 1) 优先使用用户传入的 ckpt_path（你 eval 命令里 overrides 传的那个）
    ckpt_path = usr_args.get("ckpt_path", None)
    if ckpt_path is not None and str(ckpt_path).strip() != "":
        ckpt_file = str(ckpt_path)

    # 2) 否则：nash 的保存目录是 task-seed（匹配你训练出来的 place_burger_fries-42）
    elif usr_args.get("ckpt_setting", "") == "nash":
        ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"

    # 3) 其它 setting 仍沿用原来的规则
    else:
        ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2  # 2 gripper

    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']

    print(f"[DEBUG] Initializing DP model... ckpt={ckpt_file} n_obs={n_obs_steps} n_action={n_action_steps}")
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    # print("[DEBUG] eval start")
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    # print("[DEBUG] calling model.get_action")
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()
