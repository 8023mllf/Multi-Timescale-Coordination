from pathlib import Path

from bigym.action_modes import JointPositionActionMode, PelvisDof
from demonstrations.demo_store import DemoStore, DemoNotFoundError
from demonstrations.utils import Metadata
from tools.shared.utils import ENVIRONMENTS


demos_amount = -1
output_directory = Path(__file__).resolve().parent / "demo"

action_modes = [
    JointPositionActionMode(
        absolute=True,
        floating_base=True,
        floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
    ),
    JointPositionActionMode(absolute=True, floating_base=True),
]
demo_store = DemoStore()

for env_cls in ENVIRONMENTS.values():
    for action_mode in action_modes:
        env = env_cls(action_mode=action_mode)
        try:
            demos = demo_store.get_demos(
                metadata=Metadata.from_env(env, is_lightweight=True),
                amount=demos_amount,
            )
            for demo in demos:
                demo.save(output_directory / demo.metadata.filename)
            break
        except DemoNotFoundError:
            continue
