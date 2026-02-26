import enum
import re
from pathlib import Path
from typing import Type, Optional

import mujoco
import numpy as np
import mediapy as media
from PIL import Image

from bigym.action_modes import JointPositionActionMode
from bigym.bigym_env import BiGymEnv
from demonstrations.demo import Demo
from demonstrations.demo_converter import DemoConverter

RGB_RENDER_MODE = "rgb_array"
VIDEO_FPS = 30


class Resolution(enum.Enum):
    SD = (480, 640)
    SD_SQUARE = (480, 480)
    HD = (720, 1280)
    HD_SQUARE = (720, 720)
    UHD = (2160, 3840)
    UHD_SQUARE = (2160, 3840)


class PreviewRenderer:
    def render(
        self,
        env_cls: Type[BiGymEnv],
        resolution: Resolution,
        camera: mujoco.MjvCamera,
        rendering_options: mujoco.MjvOption = mujoco.MjvOption(),
        file_name: Optional[str] = None,
        seed: Optional[int] = 0,
        output_dir: Optional[Path] = None,
    ):
        output_dir = output_dir or self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        if file_name is None:
            file_name = (
                f"task_preview_{self._env_name(env_cls)}"
                f"@{resolution.value[0]}x{resolution.value[1]}.png"
            )

        env = env_cls(
            render_mode=RGB_RENDER_MODE,
            action_mode=JointPositionActionMode(floating_base=True),
        )

        h, w = resolution.value
        env.mojo.physics.model.vis.global_.offwidth = w
        env.mojo.physics.model.vis.global_.offheight = h

        env.reset(seed=seed)
        for _ in range(100):
            env.step(env.action_space.sample() * 0)

        env.camera_id = -1
        viewer = env.mujoco_renderer.get_viewer(RGB_RENDER_MODE)
        viewer.cam = camera
        viewer.vopt = rendering_options

        render = env.render()
        Image.fromarray(render).save(output_dir / file_name)

    def render_demo(
        self,
        demo: Demo,
        resolution: Resolution,
        frequency: int,
        velocity: float,
        camera: mujoco.MjvCamera,
        rendering_options: mujoco.MjvOption = mujoco.MjvOption(),
        output_dir: Optional[Path] = None,
    ):
        output_dir = output_dir or self._get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        env = demo.metadata.get_env(frequency, RGB_RENDER_MODE)
        demo = DemoConverter.decimate(demo, frequency, robot=env.robot)

        h, w = resolution.value
        env.mojo.physics.model.vis.global_.offwidth = w
        env.mojo.physics.model.vis.global_.offheight = h

        env.camera_id = -1
        viewer = env.mujoco_renderer.get_viewer(RGB_RENDER_MODE)
        viewer.cam = camera
        viewer.vopt = rendering_options

        env.reset(seed=int(demo.seed))

        frames = []
        for i, timestep in enumerate(demo.timesteps):
            env.step(timestep.executed_action)
            camera.azimuth += velocity
            frames.append(env.render().astype(np.uint8))
        file_name = (
            f"{self._env_name(type(env))}_{demo.metadata.uuid}"
            f"@{resolution.value[0]}x{resolution.value[1]}_{velocity}.mp4"
        )
        media.write_video(output_dir / file_name, frames)

    @staticmethod
    def _env_name(env_cls: Type[BiGymEnv]) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", env_cls.__name__).lower()

    @staticmethod
    def _get_output_dir() -> Path:
        return Path(__file__).resolve().parent
