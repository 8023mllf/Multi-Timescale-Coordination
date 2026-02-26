from typing import Any

import numpy as np
import pytest
from mujoco_utils import mjcf_utils
from numpy.testing import assert_allclose

from bigym.action_modes import JointPositionActionMode
from bigym.bigym_env import BiGymEnv
from bigym.utils.env_health import (
    UnstableSimulationWarning,
    UnstableSimulationError,
    EnvHealth,
)
from bigym.envs.reach_target import ReachTarget

BASE_DELTA_RZ = np.radians(180)
UNSTABLE_ACTION_SPACE_MULTIPLIER = 100


def run_simulation(
    env: BiGymEnv, unstable: bool
) -> tuple[Any, float, bool, bool, dict]:
    timestep = env.reset()
    if unstable:
        env.action_space.low *= UNSTABLE_ACTION_SPACE_MULTIPLIER
        env.action_space.high *= UNSTABLE_ACTION_SPACE_MULTIPLIER
    for i in range(100):
        action = np.zeros_like(env.action_space.sample())
        if unstable:
            action[2] = BASE_DELTA_RZ
        timestep = env.step(action)
        if not env.is_healthy:
            break
    return timestep


def test_unstable_simulation_warns():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    with pytest.warns(UnstableSimulationWarning):
        run_simulation(env, unstable=True)


def test_unstable_simulation_is_truncated():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    with pytest.warns(UnstableSimulationWarning):
        _, _, _, truncate, _ = run_simulation(env, unstable=True)
        assert truncate


def test_unstable_simulation_consecutive_warnings():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    with pytest.raises(UnstableSimulationError):
        # Run unstable simulation multiple times to cause UnstableSimulationError
        for i in range(EnvHealth.CONSECUTIVE_WARNINGS_THRESHOLD):
            with pytest.warns(UnstableSimulationWarning):
                run_simulation(env, unstable=True)


def test_unstable_simulation_non_consecutive_warnings():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=False),
    )
    # Run unstable simulation 1 time less than the CONSECUTIVE_WARNINGS_THRESHOLD
    for i in range(EnvHealth.CONSECUTIVE_WARNINGS_THRESHOLD - 1):
        with pytest.warns(UnstableSimulationWarning):
            run_simulation(env, unstable=True)

    # Run stable simulation to reset counter
    run_simulation(env, unstable=False)
    env.reset()

    # Running unstable simulation now would not cause exception
    with pytest.warns(UnstableSimulationWarning):
        run_simulation(env, unstable=True)
        env.reset()

    env.close()


def test_gripper_model_is_stable():
    env = ReachTarget(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True)
    )
    env.reset()

    all_joints = mjcf_utils.safe_find_all(env.mojo.root_element.mjcf, "joint")
    # Gripper joints at the end of the gripper which tend be the most unstable part.
    # Was fixed in this commit in Mujoco Menagerie:
    # https://github.com/google-deepmind/mujoco_menagerie/commit/a9f3e331238e5ca82923dfbeb93f4589af97b011
    follower_joints = [
        joint
        for joint in all_joints
        if joint.name == "right_follower_joint" or joint.name == "left_follower_joint"
    ]
    joints_range = env.mojo.physics.bind(follower_joints).range
    # Acceptable "jitter" of the Robotiq 2F-85 fingertip joints is 10% of the range
    tolerance = np.average(np.ptp(joints_range, axis=1)) * 0.1
    for i in range(100):
        action = env.action_space.sample()
        action[-2:] = 0
        env.step(action)
        joints_qpos = env.mojo.physics.bind(follower_joints).qpos.copy()
        assert_allclose(joints_qpos, np.zeros_like(joints_qpos), rtol=0, atol=tolerance)
