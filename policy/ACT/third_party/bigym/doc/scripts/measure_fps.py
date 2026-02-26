# Without success and failure criteria checks
import time
import pickle

from bigym.action_modes import JointPositionActionMode
from tools.shared.utils import ENVIRONMENTS

envs = ENVIRONMENTS.values()
action_mode = JointPositionActionMode()

steps = int(5e3)
results = []
results_filename = "fps_results.pkl"

for env_cls in ENVIRONMENTS.values():
    print(f"Building {env_cls.__name__}...")
    env = env_cls(
        action_mode=action_mode,
    )
    ngeoms = env.mojo.physics.model.ngeom
    nmesh = env.mojo.physics.model.nmesh
    nmeshface = env.mojo.physics.model.nmeshface
    nmeshnormal = env.mojo.physics.model.nmeshnormal
    nmeshvert = env.mojo.physics.model.nmeshvert
    mesh_data = (ngeoms, nmesh, nmeshface, nmeshnormal, nmeshvert)
    env.reset()
    action = env.action_space.sample()

    print(f"Measuring {env_cls.__name__}...")

    start = time.time_ns()
    for _ in range(steps):
        env.step(action, fast=True)
    end = time.time_ns()

    result = (mesh_data, steps / ((end - start) / 1e9), env_cls.__name__)

    print(f"Result: {result}")
    print("-" * 40)

    results.append(result)

    with open(results_filename, "wb") as file:
        pickle.dump(results, file)

    # if len(results) >= 10:
    #     break

print("Results:")
for ngeoms, t, name in results:
    print(f"{name}: {t:.2f} steps/s", f"({ngeoms} geoms)")
