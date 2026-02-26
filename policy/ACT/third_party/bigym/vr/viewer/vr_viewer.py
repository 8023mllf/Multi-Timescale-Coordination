"""VRViewer renders a Mujoco environment to a VR headset using pyopenxr."""
import multiprocessing
import threading
from abc import ABC
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Type, Any, Union

import numpy as np
import xr
from gymnasium.core import ActType

from mojo import Mojo
from xr import FrameState, Posef

from bigym.action_modes import ActionMode
from bigym.robots.robot import Robot
from demonstrations.demo import TERMINATION_STEPS
from demonstrations.demo_recorder import DemoRecorder
from vr.viewer.control_profiles.control_profile import ControlProfile
from vr.viewer.controller import Controller
from vr.viewer import Side
from vr.viewer.vr_mujoco_renderer import VRMujocoRenderer
from vr.viewer.xr_context import XRContextObject

from bigym.bigym_env import BiGymEnv


class Resolution(Enum):
    """VR resolution options."""

    LQ = (540, 600)
    MQ = (900, 1000)
    HQ = (1440, 1600)


@dataclass
class VRViewerStats:
    """VR statistics."""

    is_recoding: bool = False
    time: float = 0
    reward: float = 0
    demos_counter: int = 0


EventType = Union[multiprocessing.Event, threading.Event]


class Countdown:
    """Countdown timer."""

    def __init__(self, delay: int):
        """Init."""
        self._delay = delay

    def step(self):
        """Step timer."""
        if self._delay > 0:
            self._delay -= 1

    @property
    def is_up(self) -> bool:
        """Is counter up."""
        return self._delay <= 0


class VRViewer:
    """Renders a Mujoco environment to a VR headset using pyopenxr."""

    STEPS_COUNT_FACTOR = 3

    OFFSET_THRESHOLD = 0.1
    OFFSET_DELTA = 0.01

    INFO_POS = np.array([2, 0, 1.8])

    def __init__(
        self,
        env_cls: Type[BiGymEnv],
        action_mode: ActionMode,
        control_profile_cls: Type[ControlProfile],
        resolution: Resolution = Resolution.LQ,
        demo_directory: Optional[Union[str, Path]] = None,
        robot_cls: Optional[Type[Robot]] = None,
    ):
        """Init.

        Notes:
            - Native resolution of the Valve Index is 1440x1600 per eye,
              using reduced 900:1000 resolution by default to improve performance.
        """
        self._width = resolution.value[0] * 2
        self._height = resolution.value[1]

        self._demo_recorder: DemoRecorder = DemoRecorder(demo_directory)

        # Will be assigned by decorated VrBiGymEnv
        self._controller_left: Optional[Controller] = None
        self._controller_right: Optional[Controller] = None

        vr_env_cls = self._vr_env(env_cls)
        self._env = vr_env_cls(
            render_mode="rgb_array", action_mode=action_mode, robot_cls=robot_cls
        )
        self._env.mojo.model.vis.global_.offwidth = self._width
        self._env.mojo.model.vis.global_.offheight = self._height
        self._env.reset()

        self._control_profile = control_profile_cls(self._env)
        self._renderer = VRMujocoRenderer(self._env.mojo, self._height, self._width)

        self._context: Optional[XRContextObject] = None
        self._space_offset = Posef()

        self._stats = VRViewerStats()
        self._stop_countdown: Optional[Countdown] = None

    def _vr_env(self, env_cls: Type[BiGymEnv]) -> Type[BiGymEnv]:
        """Add VR controllers to standard mujoco environment."""

        def add_controllers(mojo: Mojo):
            self._controller_left = Controller(Side.LEFT, mojo)
            self._controller_right = Controller(Side.RIGHT, mojo)

        def get_demo_recorder():
            return self._demo_recorder

        class VrBiGymEnv(env_cls, ABC):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._demo_recorder = get_demo_recorder()

            def _initialize_env(self):
                super()._initialize_env()
                add_controllers(self.mojo)

            def step(
                self, action: ActType, fast: bool = True
            ) -> tuple[Any, float, bool, bool, dict]:
                super().step(action, fast)
                timestep = ({}, self.reward, False, False, {})
                self._demo_recorder.add_timestep(timestep, action)
                return timestep

            @property
            def task_name(self) -> str:
                return self.__class__.__base__.__name__

        return VrBiGymEnv

    def run(
        self,
        exit_event: Optional[EventType] = threading.Event(),
        on_running_event: Optional[EventType] = threading.Event(),
    ):
        """Start VR viewer."""
        with XRContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                ],
            ),
        ) as self._context:
            on_running_event.set()
            self._renderer.set_context(self._context)
            self._controller_left.set_context(self._context)
            self._controller_right.set_context(self._context)
            for frame_state in self._context.frame_loop():
                self._handle_input(self._context)
                steps_count = self._predict_steps_count(frame_state)
                action = self._get_action(steps_count)
                for _ in range(steps_count):
                    self._env.step(action, fast=True)
                    if self._stop_countdown:
                        self._stop_countdown.step()
                        if self._stop_countdown.is_up:
                            self._stop_recording()
                    elif self._env.reward > 0:
                        self._stop_countdown = Countdown(TERMINATION_STEPS)
                self._render_frame(frame_state)
                if exit_event.is_set():
                    break

    def _get_action(self, steps_count: int) -> np.ndarray:
        action = self._control_profile.get_next_action(
            self._context, steps_count, self._space_offset
        )
        action = np.clip(
            action, self._env.action_space.low, self._env.action_space.high
        )
        return action

    def _handle_input(self, context: XRContextObject):
        # Control demo recoding
        if context.input.state[Side.LEFT].a_clicked:
            self._start_recording()
        if context.input.state[Side.LEFT].b_clicked:
            self._save_recording()

        # Control space offset
        input_y = context.input.state[Side.RIGHT].thumbstick_y
        if np.abs(input_y) >= self.OFFSET_THRESHOLD:
            self._space_offset.position.z += input_y * self.OFFSET_DELTA

        # Update controllers
        self._controller_left.update(self._space_offset)
        self._controller_right.update(self._space_offset)

    def _render_frame(self, frame_state: FrameState):
        self._update_stats()
        self._renderer.render(frame_state, self._space_offset)

    def _start_recording(self):
        self._stop_recording()
        self._space_offset = Posef()
        self._stop_countdown = None
        self._env.reset()
        self._control_profile.reset()
        self._demo_recorder.record(self._env, lightweight_demo=True)
        self._controller_left.vibrate()

    def _stop_recording(self):
        if self._demo_recorder.is_recording:
            self._demo_recorder.stop()
            self._controller_left.vibrate()
            self._stop_countdown = None

    def _save_recording(self):
        self._stop_recording()
        if self._demo_recorder.save_demo():
            self._stats.demos_counter += 1
            self._controller_left.vibrate()

    def _predict_steps_count(self, frame_state: FrameState) -> int:
        # Convert nanoseconds to seconds and divide by 2
        refresh_duration = frame_state.predicted_display_period / 2_000_000_000
        return (
            int(round(refresh_duration / self._env.mojo.physics.timestep()))
            * self.STEPS_COUNT_FACTOR
        )

    def _update_stats(self):
        self._stats.is_recoding = self._demo_recorder.is_recording
        self._stats.time = self._env.mojo.data.time
        self._stats.reward = self._env.reward
        self._renderer.show_stats(asdict(self._stats), self.INFO_POS)
