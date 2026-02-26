"""An example of using BiGym with state."""
from bigym.action_modes import TorqueActionMode
from bigym.envs.reach_target import ReachTarget

print("Running 1000 steps without pixels...")
env = ReachTarget(
    action_mode=TorqueActionMode(floating_base=True),
    render_mode="human",
)

print("Observation Space:")
print(env.observation_space)
print("Action Space:")
print(env.action_space)

env.reset()
for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    img = env.render()
    if i % 100 == 0:
        env.reset()
env.close()
