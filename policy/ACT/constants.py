import pathlib
import os, json

current_dir = os.path.dirname(__file__)

### Task parameters
SIM_TASK_CONFIGS_PATH = os.path.join(current_dir, "./SIM_TASK_CONFIGS.json")
with open(SIM_TASK_CONFIGS_PATH, "r") as f:
    SIM_TASK_CONFIGS = json.load(f)

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
START_ARM_POSE = [
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
    0,
    -0.96,
    1.16,
    0,
    -0.3,
    0,
    0.02239,
    -0.02239,
]

XML_DIR = (str(pathlib.Path(__file__).parent.resolve()) + "/assets/")  # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN -
                                                                                        MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN -
                                                                                        PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE)
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN -
                                                                                  MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN -
                                                                                  PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE)
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = (lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) *
                    (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE)
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = (lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) *
                    (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE)
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2

###############################################################################
# Data curation configs (for Nash / dual-dataset training)
###############################################################################
"""
FUSION_MAP 用来告诉 imitate_episodes_data_curation：
某个主任务是否需要和一个额外任务（比如 2x 下采样版本）一起训练。

约定：
    - key:   主任务名（必须出现在 SIM_TASK_CONFIGS 里）
    - value: {'extra_task': 对应额外任务名}，extra_task 同样必须在 SIM_TASK_CONFIGS 中有配置

脚本里会这么用：
    if task_name in FUSION_MAP:
        extra_task = FUSION_MAP[task_name]['extra_task']
        dataset_dir2 = SIM_TASK_CONFIGS[extra_task]['dataset_dir']
        ... 用 load_data_dual 加载双目录 ...
    else:
        ... 用 load_data 单目录 ...
"""

FUSION_MAP = {
    # === 你当前在 RoboTwin 下用的 beat_block_hammer 任务 ===
    # 前提：你在 SIM_TASK_CONFIGS.json 里有这两个条目：
    #   "sim_beat_block_hammer": { "dataset_dir": ".../demo_clean",     ... }
    #   "sim_beat_block_hammer_2x": { "dataset_dir": ".../demo_clean_2x", ... }
    "sim_beat_block_hammer": {
        "extra_task": "sim_beat_block_hammer_2x",
    },

    "sim_open_laptop": {
        "extra_task": "sim_open_laptop_2x",
    },

    "sim_place_burger_fries": {
        "extra_task": "sim_place_burger_fries_2x",
    },

    "sim_stack_bowls_three": {
        "extra_task": "sim_stack_bowls_three_2x",
    },

    "sim_stack_blocks_two": {
        "extra_task": "sim_stack_blocks_two_2x",
    },

    "sim_handover_block": {
        "extra_task": "sim_handover_block_2x",
    }

}
