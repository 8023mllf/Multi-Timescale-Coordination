"""VR Mujoco renderer class rendering mujoco environment to VR headset."""
from typing import Optional, Any

import mujoco
import numpy as np
import xr
from mojo import Mojo
from xr import FrameState, View, Posef, Quaternionf, Vector3f

from vr.viewer import Side
from vr.viewer.full_screen_renderer import VRFullScreenRenderer
from vr.viewer.pyopenxr_to_mujoco_converter import vector_from_pyopenxr
from vr.viewer.xr_context import XRContextObject

RENDER_REFLECTIONS = False
RENDER_SHADOWS = False
RENDER_FOG = False


class Renderer(mujoco.Renderer):
    """Customized mujoco.Renderer with decreased font size."""

    def __init__(
        self,
        model: mujoco.MjModel,
        height: int = 240,
        width: int = 320,
        max_geom: int = 10000,
    ) -> None:
        """Init."""
        super().__init__(model, height, width, max_geom)
        self._mjr_context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_50.value
        )
        self._mjr_context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._mjr_context
        )


class VRMujocoRenderer:
    """VR Mujoco renderer class rendering mujoco environment to VR headset."""

    def __init__(self, mojo: Mojo, height: int, width: int):
        """Init."""
        self._mojo = mojo
        self._width = width
        self._height = height

        self._markers = []

        self._renderer = Renderer(self._mojo.model, self._height, self._width)
        self._renderer.scene.stereo = mujoco.mjtStereo.mjSTEREO_QUADBUFFERED
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(
            RENDER_REFLECTIONS
        )
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(RENDER_SHADOWS)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = int(RENDER_FOG)
        self._vr_camera = mujoco.MjvCamera()

        # Will be initialized after creation of the VR session
        self._context: Optional[XRContextObject] = None
        self._headset_renderer: Optional[VRFullScreenRenderer] = None

    @property
    def _scene(self) -> mujoco.MjvScene:
        return self._renderer.scene

    def set_context(self, context: XRContextObject):
        """Set context of VR application."""
        self._context = context
        self._headset_renderer = VRFullScreenRenderer(self._width // 2, self._height)

    def show_stats(
        self,
        info: dict[str, Any],
        pos: np.ndarray,
        spacing: np.ndarray = np.array([0, 0, -0.1]),
    ):
        """Show label with information from dictionary."""
        label_pos = pos.copy()
        for key, value in info.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            else:
                value = str(value)
            label = f"{key}: {value}"
            self.add_marker(pos=label_pos.copy(), label=label)
            label_pos += spacing

    def add_marker(self, **marker_params):
        """Add marker to scene."""
        self._markers.append(marker_params)

    def render(self, frame_state: FrameState, offset: Posef):
        """Render current state of the environment to VR headset."""
        self._renderer.update_scene(self._mojo.data, self._vr_camera)
        self._sync_mujoco_vr_cameras_with_views(self._context.input.views, offset)
        pixels = self._render_mujoco_env()
        for view_index, _ in enumerate(self._context.view_loop(frame_state)):
            self._headset_renderer.render(Side(view_index), pixels)

    def _render_mujoco_env(self) -> np.array:
        for marker in self._markers:
            self._add_marker_to_scene(marker)
        pixels = np.flipud(self._renderer.render())
        self._markers.clear()
        return pixels

    def _sync_mujoco_vr_cameras_with_views(self, views: list[View], offset: Posef):
        for camera_id, camera in enumerate(self._scene.camera):
            view = views[camera_id]
            z_near, z_far = 0.01, 50.0
            tan_left, tan_right, tan_down, tan_up = np.tan(view.fov.as_numpy())

            # Setup camera frustum
            camera.frustum_bottom = -tan_down * z_near
            camera.frustum_top = -tan_up * z_near
            camera.frustum_center = 0.5 * (tan_left + tan_right) * z_near
            camera.frustum_near = z_near
            camera.frustum_far = z_far

            # Column-major view matrix
            orientation = xr.Matrix4x4f.create_from_quaternion(view.pose.orientation)
            if offset.orientation != Quaternionf():
                orientation_offset = xr.Matrix4x4f.create_from_quaternion(
                    offset.orientation
                )
                orientation = orientation.multiply(orientation_offset)
            orientation = orientation.as_numpy()
            # Forward is the 3rd column of the view matrix - elements [8], [9], [10]
            # Up is the 2nd column of the view matrix - elements [4], [6], [5]
            # Also we have to invert forward axis, according to the documentation:
            # https://mujoco.readthedocs.io/en/stable/programming/visualization.html
            camera.forward = -vector_from_pyopenxr(orientation[8:11])
            camera.up = vector_from_pyopenxr(orientation[4:7])
            camera.pos = vector_from_pyopenxr(view.pose.position)
            if offset.position != Vector3f():
                camera.pos += offset.position.as_numpy()

    def _add_marker_to_scene(self, marker: dict):
        if self._scene.ngeom >= self._scene.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" % self._scene.maxgeom)

        g = self._scene.geoms[self._scene.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_LABEL
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self._scene.ngeom += 1
