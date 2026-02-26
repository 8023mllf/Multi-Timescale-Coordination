"""Control profile to for H1 in floating mode."""

import numpy as np
from gymnasium.core import ActType
from pyquaternion import Quaternion
from xr import Posef

from bigym.bigym_env import BiGymEnv
from vr.ik.h1_upper_body_ik import H1UpperBodyIK, Pose
from vr.viewer import Side
from vr.viewer.control_profiles.control_profile import ControlProfile
from vr.viewer.xr_context import XRContextObject


class H1Floating(ControlProfile):
    """Control profile for H1 in floating mode.

    Notes:
        - Use controller triggers to control the grippers.
        - Use the right A button to enable/disable synchronization of position.
        - Use the right B button to enable/disable synchronization of rotation.
        - Position of controllers is used as the target for the corresponding arm of H1.
    """

    POSITION_SMOOTHING = 0.01
    ROTATION_SMOOTHING = 0.01

    HMD_TO_PELVIS_OFFSET = 0.7
    HMD_PIVOT_OFFSET = np.array([0, -0.2, 0])

    def __init__(self, env: BiGymEnv):
        """Init."""
        super().__init__(env)
        self._sync_position = True
        self._sync_rotation = True
        self._ik = H1UpperBodyIK(env)

    def get_next_action(
        self, context: XRContextObject, steps_count: int, space_offset: Posef
    ) -> ActType:
        """See base."""
        trigger_left = context.input.state[Side.LEFT].trigger_value
        trigger_right = context.input.state[Side.RIGHT].trigger_value

        l_pos, l_quat = self._get_controller_pose(context, Side.LEFT, space_offset)
        r_pos, r_quat = self._get_controller_pose(context, Side.RIGHT, space_offset)
        hmd_pos, hmd_quat = self._get_hmd_pose(
            context, space_offset, self.HMD_PIVOT_OFFSET
        )

        # Toggle position and rotation sync
        if context.input.state[Side.RIGHT].a_clicked:
            self._sync_position = not self._sync_position
        if context.input.state[Side.RIGHT].b_clicked:
            self._sync_rotation = not self._sync_rotation

        pelvis = self._env.robot.pelvis
        pelvis_pose = Pose(pelvis.get_position(), Quaternion(pelvis.get_quaternion()))

        delta_pos = np.array(hmd_pos - pelvis_pose.position)
        delta_pos[2] -= self.HMD_TO_PELVIS_OFFSET
        magnitude = np.linalg.norm(delta_pos)
        if magnitude > 1:
            delta_pos /= magnitude
        delta_pos *= self.POSITION_SMOOTHING * float(self._sync_position)

        delta_quat = (
            hmd_quat
            * pelvis_pose.orientation.inverse
            * Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
        )
        delta_ypr = np.flip(np.array(delta_quat.yaw_pitch_roll))
        delta_ypr *= self.ROTATION_SMOOTHING * float(self._sync_rotation)

        control = np.zeros_like(self._env.action_space.low)
        # Control floating base
        floating_base = self._env.robot.floating_base
        base_control = []
        if floating_base:
            for delta, actuator in zip(delta_pos, floating_base.position_actuators):
                if actuator:
                    base_control.append(delta)
            for delta, actuator in zip(delta_ypr, floating_base.rotation_actuators):
                if actuator:
                    base_control.append(delta)
            control[: len(base_control)] = base_control

        # Control arms
        start_index = floating_base.dof_amount
        end_index = start_index + len(self._env.robot.limb_actuators)

        arms_qpos = np.array(self._env.robot.qpos_actuated[start_index:end_index])
        qpos_arm_left, qpos_arm_right = np.split(arms_qpos, 2)
        solution = self._ik.solve(
            pelvis_pose=pelvis_pose,
            qpos_arm_left=qpos_arm_left,
            qpos_arm_right=qpos_arm_right,
            target_pose_left=Pose(l_pos, l_quat),
            target_pose_right=Pose(r_pos, r_quat),
        )
        control[start_index:end_index] = solution

        # Control grippers
        control[-2] = np.clip(np.round(trigger_left), 0, 1)
        control[-1] = np.clip(np.round(trigger_right), 0, 1)

        return control
