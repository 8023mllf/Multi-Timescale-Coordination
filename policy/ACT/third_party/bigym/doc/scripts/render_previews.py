from pathlib import Path

import mujoco

from doc.scripts.preview_renderer import PreviewRenderer, Resolution
from tools.shared.utils import ENVIRONMENTS

camera = mujoco.MjvCamera()
camera.distance = 3.8
camera.azimuth = -65
camera.elevation = -25
camera.lookat = [0.5, 0, 1]

output_directory = Path(__file__).resolve().parent / "previews"
envs = ENVIRONMENTS.values()

for env_cls in ENVIRONMENTS.values():
    PreviewRenderer().render(
        env_cls=env_cls,
        resolution=Resolution.HD_SQUARE,
        camera=camera,
        output_dir=output_directory,
    )
