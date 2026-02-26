"""Module for handling state of VR controllers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from mojo import Mojo
from mojo.elements import Body
from pyquaternion import Quaternion
from xr import Posef

from vr.viewer.pyopenxr_to_mujoco_converter import (
    vector_from_pyopenxr,
    quaternion_from_pyopenxr,
)
from vr.viewer import Side

if TYPE_CHECKING:
    from vr.viewer.xr_context import XRContextObject

CONTROLLER_NOT_ACTIVE_POSITION = np.array([0, 0, -100])
CONTROLLER_NOT_ACTIVE_ROTATION = np.array(Quaternion().elements)

PACKAGE_PATH = Path(__file__).parent
CONTROLLER_MODELS = {
    Side.LEFT: "xmls/controller_left/controller_left.xml",
    Side.RIGHT: "xmls/controller_right/controller_right.xml",
}


class ControllerState:
    """State of the VR Controller."""

    def __init__(self):
        """Init."""
        # Inputs
        self.is_active: bool = False
        self.pose: Posef = Posef()
        self.pose_aim: Posef = Posef()
        self.trigger_click: bool = False
        self.trigger_changed: bool = False
        self.trigger_value: float = 0.0
        self.a_click: bool = False
        self.a_changed: bool = False
        self.b_click: bool = False
        self.b_changed: bool = False
        self.thumbstick_x: float = 0.0
        self.thumbstick_y: float = 0.0
        # Outputs
        self.vibration: bool = False

    @property
    def a_clicked(self):
        """True if A was clicked during this frame."""
        return self.a_click and self.a_changed

    @property
    def b_clicked(self):
        """True if B was clicked during this frame."""
        return self.b_click and self.b_changed

    @property
    def trigger_clicked(self):
        """True if Trigger was clicked during this frame."""
        return self.trigger_click and self.trigger_changed


class Controller:
    """Manages VR controller state in Mujoco."""

    def __init__(
        self,
        side: Side,
        mojo: Mojo,
    ):
        """Init."""
        model_path = PACKAGE_PATH / CONTROLLER_MODELS[side]
        self._side = side
        self._controller: Body = mojo.load_model(str(model_path))
        self._context: Optional[XRContextObject] = None
        self.update(Posef())

    def set_context(self, context: XRContextObject):
        """Assign XRContext object."""
        self._context = context

    def update(self, space_offset: Posef):
        """Update state of the controller model in Mujoco environment."""
        state = (
            self._context.input.state[self._side]
            if self._context
            else ControllerState()
        )
        if state.is_active:
            pose = state.pose
            self._controller.set_position(
                vector_from_pyopenxr(pose.position) + space_offset.position.as_numpy()
            )
            self._controller.set_quaternion(quaternion_from_pyopenxr(pose.orientation))
        else:
            self._controller.set_position(CONTROLLER_NOT_ACTIVE_POSITION)
            self._controller.set_quaternion(CONTROLLER_NOT_ACTIVE_ROTATION)

    def vibrate(self):
        """Activate controller's vibration."""
        if self._context:
            self._context.input.state[self._side].vibration = True
