"""Control profile for arbitrary robot in floating mode."""

import numpy as np
from gymnasium.core import ActType
from pyquaternion import Quaternion
from xr import Posef

from bigym.bigym_env import BiGymEnv
from bigym.utils.robot_highlighter import RobotHighlighter
from vr.viewer import Side
from vr.viewer.control_profiles.control_profile import ControlProfile
from vr.viewer.xr_context import XRContextObject


class UniversalFloating(ControlProfile):
    """Control arbitrary robot in floating mode.

    Notes:
        - Use left joysticks to control base or selected joint
        - Use left trigger to control base/joints
        - Use right trigger to select controlled joint
    """

    _SENS_POS = 0.5
    _SENS_ROT = 0.25
    _SENS_DIRECT = 2
    _SENS_THRESHOLD = 0.5

    _HIGHLIGHT_TINT = np.array([0, 0.5, 0, 1])

    def __init__(self, env: BiGymEnv):
        """Init."""
        super().__init__(env)

        config = self._env.robot.config
        self._fwd_delta = self._SENS_POS * max(
            config.floating_base.delta_range_position
        )
        self._turn_delta = (
            -1 * self._SENS_ROT * max(config.floating_base.delta_range_rotation)
        )
        self._joint_delta = self._SENS_DIRECT * max(config.delta_range)

        self._joint_index = 0
        self._joints_count = len(self._env.robot.qpos_actuated) - len(
            self._env.robot.floating_base.qpos
        )

        self._highlighter = RobotHighlighter(self._env.robot, self._env.mojo)

        self._control_base = True
        self._indicate_control_mode()

    def get_next_action(
        self, context: XRContextObject, steps_count: int, space_offset: Posef
    ) -> ActType:
        """See base."""
        left_x = self._joystick_value(context.input.state[Side.LEFT].thumbstick_x)
        left_y = self._joystick_value(context.input.state[Side.LEFT].thumbstick_y)
        if context.input.state[Side.LEFT].trigger_clicked:
            self._toggle_control_mode()
        if context.input.state[Side.RIGHT].trigger_clicked:
            self._toggle_target_joint()

        if self._control_base:
            control = self._get_base_control(left_x, left_y, steps_count)
        else:
            control = self._get_joints_control(left_y, steps_count)
        return control

    def _get_base_control(self, turn: float, forward: float, steps: int) -> np.ndarray:
        control = self._env.action
        forward *= self._fwd_delta
        turn *= self._turn_delta
        pelvis_quat = Quaternion(self._env.robot.pelvis.get_quaternion())
        delta_position = pelvis_quat.rotate(np.array([forward, 0, 0])) / steps
        delta_rotation = np.array([0, 0, turn]) / steps
        base = self._env.robot.floating_base
        base_control = []
        for i, actuator in enumerate(base.position_actuators):
            if not actuator:
                continue
            base_control.append(delta_position[i])
        for i, actuator in enumerate(base.rotation_actuators):
            if not actuator:
                continue
            base_control.append(delta_rotation[i])
        control[: len(base_control)] = base_control
        return control

    def _get_joints_control(self, value: float, steps: int) -> np.ndarray:
        # Fix base
        base_dofs = self._env.robot.floating_base.dof_amount
        control = self._env.action
        control[:base_dofs] = np.zeros(base_dofs)
        # Control selected joint
        control_index = base_dofs + self._joint_index
        is_gripper = ((len(control) - 1) - control_index) < len(
            self._env.robot.grippers
        )
        if is_gripper and value != 0:
            control[control_index] = 0 if value < 0 else 1
        else:
            delta = value * self._joint_delta / steps
            control[control_index] += delta
        return control

    def _joystick_value(self, value: float) -> float:
        value = np.clip(value, -1, 1)
        return value if np.abs(value) >= self._SENS_THRESHOLD else 0

    def _toggle_control_mode(self):
        self._control_base = not self._control_base
        self._indicate_control_mode()

    def _toggle_target_joint(self):
        if self._control_base:
            return
        self._joint_index += 1
        if self._joint_index >= self._joints_count:
            self._joint_index = 0
        self._indicate_control_mode()

    def _indicate_control_mode(self):
        self._highlighter.reset()
        if not self._control_base:
            self._highlighter.highlight(self._joint_index, self._HIGHLIGHT_TINT)
