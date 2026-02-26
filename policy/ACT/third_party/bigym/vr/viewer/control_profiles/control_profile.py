"""Abstract base class for defining control profiles."""
from abc import ABC, abstractmethod

import numpy as np
from gymnasium.core import ActType
from pyquaternion import Quaternion
from xr import Posef

from bigym.bigym_env import BiGymEnv
from vr.viewer import Side
from vr.viewer.pyopenxr_to_mujoco_converter import (
    vector_from_pyopenxr,
    quaternion_from_pyopenxr,
)
from vr.viewer.xr_context import XRContextObject


class ControlProfile(ABC):
    """Abstract base class for defining control profiles."""

    def __init__(self, env: BiGymEnv):
        """Init."""
        self._env = env

    @abstractmethod
    def get_next_action(
        self,
        context: XRContextObject,
        steps_count: int,
        space_offset: Posef,
    ) -> ActType:
        """Get the next action.

        :param context: XR context object to access current input.
        :param steps_count: Amount of physical steps to be taken.
            Divide delta actions by this value in order to keep actions
            consistent regardless of the current frame rate.
        :param space_offset: Virtual space offset.
        """
        pass

    def reset(self):
        """Custom reset behaviour, called on environment reset."""
        pass

    @staticmethod
    def _get_controller_pose(
        context: XRContextObject, side: Side, offset: Posef
    ) -> tuple[np.ndarray, Quaternion]:
        pose = context.input.state[side].pose_aim
        pos = vector_from_pyopenxr(pose.position) + offset.position.as_numpy()
        quat = Quaternion(quaternion_from_pyopenxr(pose.orientation))
        return pos, quat

    @staticmethod
    def _get_hmd_pose(
        context: XRContextObject, offset: Posef, pivot_offset: np.ndarray = np.zeros(3)
    ) -> tuple[np.ndarray, Quaternion]:
        pose = context.input.hmd_pose
        quat = Quaternion(quaternion_from_pyopenxr(pose.orientation))
        pos = vector_from_pyopenxr(pose.position) + offset.position.as_numpy()
        pos += quat.rotate(pivot_offset)
        return pos, quat
