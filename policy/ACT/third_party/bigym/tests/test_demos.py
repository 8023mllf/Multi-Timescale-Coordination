"""Test demonstration collection."""
import copy
import pytest
import tempfile

from numpy.testing import assert_allclose

from bigym.action_modes import TorqueActionMode, JointPositionActionMode
from bigym.bigym_env import BiGymEnv
from bigym.envs.move_plates import MovePlate
from bigym.envs.reach_target import ReachTarget
from bigym.envs.manipulation import StackBlocks
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from demonstrations.const import ACTION_KEY
from demonstrations.demo import Demo
from demonstrations.demo_recorder import DemoRecorder
from demonstrations.demo_converter import DemoConverter


def assert_timesteps_equal(d1: Demo, d2: Demo, atol=1e-6) -> None:
    assert len(d1.timesteps) == len(d2.timesteps)
    for t1, t2 in zip(d1.timesteps, d2.timesteps):
        assert_allclose(t1.executed_action, t2.executed_action, atol=atol)
        assert t1.termination == t2.termination
        assert t1.truncation == t2.truncation
        assert t1.reward == t2.reward
        for key, val in t1.observation.items():
            assert_allclose(val, t2.observation[key], atol=atol)
        for key, val in t1.info.items():
            assert_allclose(val, t2.info[key], atol=atol)


@pytest.mark.parametrize(
    "env_class",
    [ReachTarget, MovePlate, StackBlocks],
)
@pytest.mark.parametrize(
    "action_mode",
    [JointPositionActionMode, TorqueActionMode],
)
class TestDemos:
    """Test demonstration collection."""

    @staticmethod
    def record_demo(env: BiGymEnv, recorder: DemoRecorder, length: int = 10):
        env.reset()
        recorder.record(env)
        for _ in range(length):
            action = env.action_space.sample()
            timestep = env.step(action)
            recorder.add_timestep(timestep, action)
        recorder.stop()

    @staticmethod
    def assert_replay_observations(env: BiGymEnv, demo: Demo, atol=1e-6) -> None:
        env.reset(seed=demo.seed)
        for timestep in demo.timesteps:
            action = timestep.executed_action
            obs, _, _, _, _ = env.step(action)
            for key, val in timestep.observation.items():
                assert_allclose(val, obs[key], atol=atol), f"Key: {key}"

    def test_save_and_load_demo(self, env_class, action_mode):
        env: BiGymEnv = env_class(action_mode=action_mode())
        env.reset()
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DemoRecorder(temp_dir)
            self.record_demo(env, recorder)
            filepath = recorder.save_demo()
            demo = Demo.from_safetensors(filepath)
            self.assert_replay_observations(env, demo)


def test_action_conversion():
    absolute_env: BiGymEnv = ReachTarget(
        action_mode=JointPositionActionMode(absolute=True)
    )
    delta_env: BiGymEnv = ReachTarget(
        action_mode=JointPositionActionMode(absolute=False)
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        recorder = DemoRecorder(temp_dir)
        TestDemos.record_demo(absolute_env, recorder, length=100)
        demo = recorder.demo
        delta_demo = DemoConverter.absolute_to_delta(copy.deepcopy(demo))
        delta_demo = DemoConverter.create_demo_in_new_env(delta_demo, delta_env)
        TestDemos.assert_replay_observations(absolute_env, demo)
        TestDemos.assert_replay_observations(delta_env, delta_demo)


def test_demo_env_conversion():
    old_env: BiGymEnv = ReachTarget(action_mode=JointPositionActionMode(absolute=True))
    with tempfile.TemporaryDirectory() as temp_dir:
        recorder = DemoRecorder(temp_dir)
        TestDemos.record_demo(old_env, recorder, length=10)
        old_demo = recorder.demo
        for timestep in old_demo.timesteps:
            assert "rgb_head" not in dict(timestep.observation)
            assert "rgb_right_wrist" not in dict(timestep.observation)
            assert "rgb_left_wrist" not in dict(timestep.observation)
        new_env: BiGymEnv = ReachTarget(
            action_mode=JointPositionActionMode(absolute=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name=name, resolution=(32, 32))
                    for name in ["head", "right_wrist", "left_wrist"]
                ],
            ),
        )
        new_demo = DemoConverter.create_demo_in_new_env(old_demo, new_env)
        for timestep in new_demo.timesteps:
            obs = timestep.observation
            assert "rgb_head" in dict(obs)
            assert "rgb_right_wrist" in dict(obs)
            assert "rgb_left_wrist" in dict(obs)
            assert obs["rgb_head"].shape == (3, 32, 32)
            assert obs["rgb_right_wrist"].shape == (3, 32, 32)
            assert obs["rgb_left_wrist"].shape == (3, 32, 32)
        for old_timestep, new_timestep in zip(old_demo.timesteps, new_demo.timesteps):
            assert_allclose(
                old_timestep.executed_action, new_timestep.executed_action, atol=1e-6
            )
        assert new_demo.seed == old_demo.seed

        for t1, t2 in zip(old_demo.timesteps, new_demo.timesteps):
            assert_allclose(t1.executed_action, t2.executed_action, atol=1e-6)
            assert t1.termination == t2.termination
            assert t1.truncation == t2.truncation
            assert t1.reward == t2.reward


def test_save_and_load_lightweight_demo():
    env: BiGymEnv = ReachTarget(action_mode=JointPositionActionMode())
    with tempfile.TemporaryDirectory() as temp_dir:
        recorder = DemoRecorder(temp_dir)
        recorder.record(env, lightweight_demo=True)
        for _ in range(100):
            action = env.action_space.sample()
            timestep = env.step(action)
            recorder.add_timestep(timestep, action)
        recorder.stop()
        before_saving = recorder.demo
        filepath = recorder.save_demo()
        after_saving = Demo.from_safetensors(filepath)
        assert_timesteps_equal(before_saving, after_saving)


def test_recreation_from_lightweight_demo():
    env: BiGymEnv = ReachTarget(
        action_mode=JointPositionActionMode(absolute=True),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(name=name, resolution=(32, 32))
                for name in ["head", "right_wrist", "left_wrist"]
            ],
        ),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        heavy_recorder = DemoRecorder(temp_dir)
        lightweight_recorder = DemoRecorder(temp_dir)
        env.reset()
        heavy_recorder.record(env)
        lightweight_recorder.record(env, lightweight_demo=True)
        for _ in range(10):
            action = env.action_space.sample()
            timestep = env.step(action)
            heavy_recorder.add_timestep(timestep, action)
            lightweight_recorder.add_timestep(timestep, action)
        heavy_recorder.stop()
        lightweight_recorder.stop()

        heavy_demo = heavy_recorder.demo
        lightweight_demo = lightweight_recorder.demo
        for timestep in lightweight_demo.timesteps:
            assert timestep.observation == {}
            assert timestep.reward is None
            assert list(timestep.info.keys()) == [ACTION_KEY]

        recreated_demo = DemoConverter.create_demo_in_new_env(lightweight_demo, env)

        assert lightweight_demo.uuid == recreated_demo.uuid

        for timestep in recreated_demo.timesteps:
            obs = timestep.observation
            assert "rgb_head" in dict(obs)
            assert "rgb_right_wrist" in dict(obs)
            assert "rgb_left_wrist" in dict(obs)
            assert obs["rgb_head"].shape == (3, 32, 32)
            assert obs["rgb_right_wrist"].shape == (3, 32, 32)
            assert obs["rgb_left_wrist"].shape == (3, 32, 32)

        # The pixel values should be deterministic when using the nvidia-driver-535.
        # Values could be non-deterministic due to different graphical driver
        # configurations (e.g. using EGL instead of GLFW).
        assert_timesteps_equal(heavy_demo, recreated_demo)


def test_long_running_demo():
    env: BiGymEnv = ReachTarget(action_mode=JointPositionActionMode())
    with tempfile.TemporaryDirectory() as temp_dir:
        recorder = DemoRecorder(temp_dir)
        TestDemos.record_demo(env, recorder, length=1000)
        filepath = recorder.save_demo()
        demo = Demo.from_safetensors(filepath)
        assert len(demo.timesteps) == 1000
        TestDemos.assert_replay_observations(env, demo)
