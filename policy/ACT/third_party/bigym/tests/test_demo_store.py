"""Test demonstration storing system."""
import tempfile

import pytest
from pathlib import Path
from typing import Optional

from bigym.action_modes import (
    JointPositionActionMode,
    TorqueActionMode,
    ActionMode,
)
from bigym.bigym_env import BiGymEnv
from bigym.envs.reach_target import ReachTarget
from bigym.envs.move_plates import MovePlate
from bigym.envs.manipulation import StackBlocks
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from demonstrations.demo import Demo, LightweightDemo
from demonstrations.demo_recorder import DemoRecorder
from demonstrations.demo_store import (
    DemoStore,
    DemoNotFoundError,
    TooManyDemosRequestedError,
)
from demonstrations.utils import ObservationMode, Metadata
from demonstrations.const import SAFETENSORS_SUFFIX

from tests.test_demos import assert_timesteps_equal

ENV_CLASSES = [ReachTarget, MovePlate, StackBlocks]
ACTION_MODES = [JointPositionActionMode, TorqueActionMode]

NUM_DEMOS_PER_ENV = 2

STATE_CONFIG = ObservationConfig()
PIXEL_CONFIG = ObservationConfig(
    cameras=[CameraConfig(name="head", rgb=True, depth=False, resolution=(4, 4))]
)


@pytest.fixture()
def temp_demo_store():
    with tempfile.TemporaryDirectory() as temp_dir:
        demo_store = DemoStore(Path(temp_dir))
        demo_store.cached = True
        yield demo_store


class TestDemoStore:
    """Test demonstration storing system."""

    @staticmethod
    def record_demo(
        env: BiGymEnv,
        recorder: DemoRecorder,
        length: int = 2,
        seed: Optional[int] = None,
        is_lightweight: bool = False,
    ):
        env.reset(seed=seed)
        recorder.record(env, lightweight_demo=is_lightweight)
        for _ in range(length):
            action = env.action_space.sample()
            timestep = env.step(action)
            recorder.add_timestep(timestep, action)
        recorder.stop()

    @staticmethod
    def generate_demos(
        env_classes: list[type[BiGymEnv]] = ENV_CLASSES,
        action_modes: list[type[ActionMode]] = ACTION_MODES,
        obs_modes: list[ObservationMode] = list(ObservationMode),
        num_demos_per_env: int = NUM_DEMOS_PER_ENV,
    ) -> dict[int, Demo]:
        demos = {}
        recorder = DemoRecorder()
        seed = 0
        for env_class in env_classes:
            for action_mode in action_modes:
                for obs_mode in obs_modes:
                    observation_config = (
                        PIXEL_CONFIG
                        if obs_mode == ObservationMode.Pixel
                        else STATE_CONFIG
                    )
                    is_lightweight = obs_mode == ObservationMode.Lightweight
                    for _ in range(num_demos_per_env):
                        env: BiGymEnv = env_class(
                            action_mode=action_mode(),
                            observation_config=observation_config,
                        )
                        TestDemoStore.record_demo(
                            env, recorder, seed=seed, is_lightweight=is_lightweight
                        )
                        demo = recorder.demo
                        demos[seed] = demo
                        seed += 1
                    seed += 1
                seed += 1
            seed += 1
        return demos

    def test_upload_and_download_of_a_demo(self, temp_demo_store):
        recorder = DemoRecorder()
        env = ReachTarget(action_mode=JointPositionActionMode())
        self.record_demo(env, recorder, seed=42)
        demo_to_store = recorder.demo
        temp_demo_store.cache_demo(demo_to_store)
        demos_from_store = temp_demo_store.get_demos(Metadata.from_env(env))
        assert len(demos_from_store) == 1
        demo_from_store = demos_from_store[0]
        assert_timesteps_equal(demo_to_store, demo_from_store)

    @staticmethod
    def _test_upload_and_download_multiple_demos(
        demo_store: DemoStore,
        env_classes: list[type[BiGymEnv]] = ENV_CLASSES,
        action_modes: list[type[ActionMode]] = ACTION_MODES,
        obs_modes: list[ObservationMode] = list(ObservationMode),
        num_demos_per_env: int = NUM_DEMOS_PER_ENV,
    ):
        demos_to_cache = TestDemoStore.generate_demos(
            env_classes=env_classes,
            action_modes=action_modes,
            obs_modes=obs_modes,
            num_demos_per_env=num_demos_per_env,
        )
        for demo in list(demos_to_cache.values()):
            demo_store.cache_demo(demo)

        for env_class in env_classes:
            for action_mode in action_modes:
                for obs_mode in obs_modes:
                    observation_config = (
                        PIXEL_CONFIG
                        if obs_mode == ObservationMode.Pixel
                        else STATE_CONFIG
                    )
                    metadata = Metadata.from_env_cls(
                        env_class,
                        action_mode,
                        obs_mode=obs_mode,
                        observation_config=observation_config,
                        action_mode_absolute=None
                        if action_mode == TorqueActionMode
                        else False,
                    )
                    demos_from_store = demo_store.get_demos(metadata)
                    if obs_mode == ObservationMode.Lightweight:
                        assert len(demos_from_store) == num_demos_per_env * len(
                            obs_modes
                        )
                    else:
                        assert len(demos_from_store) == num_demos_per_env
                    for demo_from_store in demos_from_store:
                        demo_to_store = demos_to_cache[demo_from_store.metadata.seed]
                        if obs_mode == ObservationMode.Lightweight:
                            demo_to_store = LightweightDemo.from_demo(demo_to_store)
                        assert_timesteps_equal(demo_from_store, demo_to_store)

    def test_upload_and_download_of_demos(self, temp_demo_store):
        self._test_upload_and_download_multiple_demos(
            temp_demo_store,
            env_classes=[ReachTarget],
            action_modes=[JointPositionActionMode],
            obs_modes=[ObservationMode.Lightweight],
            num_demos_per_env=2,
        )

    def test_upload_and_download_of_demos_with_multiple_envs(self, temp_demo_store):
        self._test_upload_and_download_multiple_demos(
            temp_demo_store,
            env_classes=ENV_CLASSES,
            action_modes=[JointPositionActionMode],
            obs_modes=[ObservationMode.Lightweight],
            num_demos_per_env=1,
        )

    def test_upload_and_download_of_demos_with_multiple_action_modes(
        self, temp_demo_store
    ):
        self._test_upload_and_download_multiple_demos(
            temp_demo_store,
            env_classes=[ReachTarget],
            action_modes=ACTION_MODES,
            obs_modes=[ObservationMode.Lightweight],
            num_demos_per_env=1,
        )

    def test_upload_and_download_of_demos_with_multiple_obs_modes(
        self, temp_demo_store
    ):
        self._test_upload_and_download_multiple_demos(
            temp_demo_store,
            env_classes=[ReachTarget],
            action_modes=[JointPositionActionMode],
            obs_modes=list(ObservationMode),
            num_demos_per_env=1,
        )

    def test_correct_file_structure(self, temp_demo_store):
        # Load the demos from the test_data folder and upload them to the cloud
        path = Path(__file__).parent / "data/safetensors"
        demo_files = list(path.glob(f"*{SAFETENSORS_SUFFIX}"))
        for demo_file in demo_files:
            temp_demo_store._cache_demo_file(demo_file)

        for env_class in ENV_CLASSES:
            for action_mode in ACTION_MODES:
                for obs_mode in list(ObservationMode):
                    observation_config = (
                        PIXEL_CONFIG
                        if obs_mode == ObservationMode.Pixel
                        else STATE_CONFIG
                    )
                    metadata = Metadata.from_env_cls(
                        env_class,
                        action_mode,
                        obs_mode=obs_mode,
                        observation_config=observation_config,
                        action_mode_absolute=None
                        if action_mode == TorqueActionMode
                        else False,
                    )
                    paths = temp_demo_store.list_demo_paths(metadata)
                    assert len(paths) > 0
                    path = paths[0]
                    expected_path = (
                        temp_demo_store._cache_path
                        / metadata.env_name
                        / metadata.environment_data.action_mode_description
                        / obs_mode.value
                    )
                    if obs_mode == ObservationMode.Pixel:
                        expected_path /= metadata.environment_data.camera_description
                    assert path.parent == expected_path

    def test_get_demo_with_new_observations(self, temp_demo_store):
        env = ReachTarget(
            action_mode=JointPositionActionMode(absolute=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name=name, resolution=(4, 4))
                    for name in ["head", "right_wrist", "left_wrist"]
                ],
            ),
        )
        env.reset()
        original_metadata = Metadata.from_env(env)
        heavy_recorder = DemoRecorder()
        lightweight_recorder = DemoRecorder()
        heavy_recorder.record(env)
        lightweight_recorder.record(env, lightweight_demo=True)
        for _ in range(5):
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

        temp_demo_store.cache_demo(lightweight_demo)
        demos_from_store = temp_demo_store.get_demos(original_metadata)
        assert temp_demo_store.light_demo_exists(lightweight_demo.metadata)
        assert len(demos_from_store) == 1
        recreated_demo = demos_from_store[0]

        for timestep in recreated_demo.timesteps:
            obs = timestep.observation
            assert "rgb_head" in dict(obs)
            assert "rgb_right_wrist" in dict(obs)
            assert "rgb_left_wrist" in dict(obs)
            assert obs["rgb_head"].shape == (3, 4, 4)
            assert obs["rgb_right_wrist"].shape == (3, 4, 4)
            assert obs["rgb_left_wrist"].shape == (3, 4, 4)

        assert_timesteps_equal(heavy_demo, recreated_demo)

    def test_retrieve_n_demos(self, temp_demo_store):
        env = ReachTarget(
            action_mode=JointPositionActionMode(absolute=True),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name=name, resolution=(4, 4))
                    for name in ["head", "right_wrist", "left_wrist"]
                ],
            ),
        )
        metadata = Metadata.from_env(env)
        recorder = DemoRecorder()
        demos = []
        for i in range(10):
            env.reset()
            recorder.record(env)
            action = env.action_space.sample()
            timestep = env.step(action)
            recorder.add_timestep(timestep, action)
            recorder.stop()
            demos.append(recorder.demo)

        for demo in demos:
            temp_demo_store.cache_demo(demo)

        for i in range(0, 11, 2):
            demos_from_store = temp_demo_store.get_demos(metadata, amount=i)
            assert len(demos_from_store) == i

    def test_implicit_saving_of_lightweight_demos(self, temp_demo_store):
        demo = _generate_simple_demo()
        temp_demo_store.cache_demo(demo)
        assert temp_demo_store.light_demo_exists(demo.metadata)

    def test_exception_thrown_if_demos_do_not_exist(self, temp_demo_store):
        metadata = Metadata.from_env_cls(
            ReachTarget,
            JointPositionActionMode,
            obs_mode=ObservationMode.Lightweight,
        )
        with pytest.raises(DemoNotFoundError):
            temp_demo_store.get_demos(metadata)

    def test_exception_thrown_if_lightweight_demos_do_not_exist(self, temp_demo_store):
        metadata = Metadata.from_env_cls(
            ReachTarget,
            JointPositionActionMode,
            obs_mode=ObservationMode.State,
        )
        with pytest.raises(DemoNotFoundError):
            temp_demo_store.get_demos(metadata)

    def test_exception_thrown_if_too_many_demos_requested(self, temp_demo_store):
        demo = _generate_simple_demo()
        temp_demo_store.cache_demo(demo)
        with pytest.raises(TooManyDemosRequestedError):
            temp_demo_store.get_demos(demo.metadata, amount=1000)


def _generate_simple_demo():
    recorder = DemoRecorder()
    env = ReachTarget(action_mode=JointPositionActionMode())
    TestDemoStore.record_demo(env, recorder, seed=42)
    return recorder.demo
