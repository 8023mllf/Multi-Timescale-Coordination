# Without success and failure criteria checks
import time
import pickle

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget
from bigym.utils.observation_config import CameraConfig, ObservationConfig
from tools.shared.utils import ENVIRONMENTS

envs = ENVIRONMENTS.values()
action_mode = JointPositionActionMode()

steps = int(5e3)
results = []
results_filename = "fps_cam_results.pkl"

RGB_RESOLUTION = 84
CAMERAS = [
    CameraConfig(
        name="head",
        rgb=True,
        depth=True,
        resolution=(RGB_RESOLUTION, RGB_RESOLUTION),
    ),
    CameraConfig(
        name="right_wrist",
        rgb=True,
        depth=True,
        resolution=(RGB_RESOLUTION, RGB_RESOLUTION),
    ),
    CameraConfig(
        name="left_wrist",
        rgb=True,
        depth=True,
        resolution=(RGB_RESOLUTION, RGB_RESOLUTION),
    ),
]

for i in [0, 1, 2, 3]:
    print(f"Building with {i} cameras on ReachTarget...")
    env = ReachTarget(
        action_mode=action_mode,
        observation_config=ObservationConfig(cameras=CAMERAS[:i]),
        render_mode=None,
    )
    env.reset()
    action = env.action_space.sample()

    print("Measuring...")

    start = time.time_ns()
    for _ in range(steps):
        env.step(action)
    end = time.time_ns()

    result = (i, steps / ((end - start) / 1e9))

    print(f"Result: {result}")
    print("-" * 40)

    results.append(result)

    with open(results_filename, "wb") as file:
        pickle.dump(results, file)

print("Results:")
for ncams, t in results:
    print(f"{t:.2f} steps/s", f"({ncams} cameras)")
