"""Customized pyopenxr ContextObject."""
import platform
from typing import Optional

import xr
from xr import (
    InstanceCreateInfo,
    SessionCreateInfo,
    ReferenceSpaceCreateInfo,
    ViewConfigurationType,
    EnvironmentBlendMode,
    FormFactor,
)

from vr.viewer.xr_input import XRInput

ALWAYS_DESTROY_INSTANCE_ON_EXIT = True


class XRContextObject(xr.ContextObject):
    """Customized pyopenxr ContextObject.

    Notes:
        - Handles update loop of the XRInput object.
        - Fixes the issue of "hanging" when calling `xr.destroy_instance`.
    """

    def __init__(
        self,
        instance_create_info=InstanceCreateInfo(),
        session_create_info=SessionCreateInfo(),
        reference_space_create_info=ReferenceSpaceCreateInfo(),
        view_configuration_type=ViewConfigurationType.PRIMARY_STEREO,
        environment_blend_mode=EnvironmentBlendMode.OPAQUE,
        form_factor=FormFactor.HEAD_MOUNTED_DISPLAY,
    ):
        """Init."""
        super().__init__(
            instance_create_info,
            session_create_info,
            reference_space_create_info,
            view_configuration_type,
            environment_blend_mode,
            form_factor,
        )
        # Will be initialized later in __enter__
        self.input: Optional[XRInput] = None

    def __enter__(self):
        """Initializes XRInput upon entering the context."""
        enter_result = super().__enter__()
        self.input = XRInput(self)
        return enter_result

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleans up XR resources upon exiting the context.

        Contains fix to prevent application hang on Linux: https://github.com/ValveSoftware/SteamVR-for-Linux/issues/422.
        """
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None
        if self.instance is not None:
            # Workaround to prevent hang
            if ALWAYS_DESTROY_INSTANCE_ON_EXIT or platform.system() != "Linux":
                xr.destroy_instance(self.instance)
            self.instance = None

    def frame_loop(self):
        """Runs the frame loop and updates XR input."""
        for frame_state in super().frame_loop():
            self.input.update(frame_state.predicted_display_time)
            yield frame_state
