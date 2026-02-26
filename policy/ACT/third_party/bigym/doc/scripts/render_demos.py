import itertools
from pathlib import Path

import mujoco

from demonstrations.demo import Demo
from doc.scripts.preview_renderer import PreviewRenderer, Resolution
from tools.shared.utils import get_demos_in_dir

fps = 30
script_directory = Path(__file__).resolve().parent
demos_directory = script_directory / "demo"
output_directory = script_directory / "recordings"

resolutions = [Resolution.HD_SQUARE]
velocities = [0.25, -0.25]


def get_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.distance = 3.8
    camera.azimuth = -65
    camera.elevation = -25
    camera.lookat = [0.5, 0, 1]
    return camera


for file in get_demos_in_dir(demos_directory):
    demo = Demo.from_safetensors(file)
    for resolution, velocity in itertools.product(resolutions, velocities):
        PreviewRenderer().render_demo(
            demo=demo,
            resolution=resolution,
            camera=get_camera(),
            frequency=fps,
            velocity=velocity,
            output_dir=output_directory,
        )
