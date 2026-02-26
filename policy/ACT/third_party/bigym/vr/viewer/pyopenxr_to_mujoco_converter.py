"""Converts vectors and quaternions from pyopenxr to mujoco space."""
import numpy as np
import xr
from pyquaternion import Quaternion
from xr import Vector3f, Quaternionf


def vector_from_pyopenxr(xr_vector: [Vector3f, np.array]) -> np.ndarray:
    """Convert pyopenxr vector to mujoco space.

    To convert from pyopenxr to mujoco, a 90-degree rotation along the X-axis
    has to be applied, i.e., multiplication by the following offset matrix:

    | 1 0  0 |
    | 0 0 -1 |
    | 0 1  0 |

    mujoco_vector = [xr_vector[0], -xr_vector[2], xr_vector[1]]
    """
    if isinstance(xr_vector, Vector3f):
        xr_vector = xr_vector.as_numpy()
    return np.array([xr_vector[0], -xr_vector[2], xr_vector[1]])


def quaternion_from_pyopenxr(xr_quaternion: Quaternionf) -> np.ndarray:
    """Convert pyopenxr quaternion to mujoco space."""
    xr_quaternion = Quaternion(
        xr_quaternion.w, xr_quaternion.x, xr_quaternion.y, xr_quaternion.z
    )
    xr_quaternion = Quaternion(axis=[1, 0, 0], degrees=90).rotate(xr_quaternion)
    return xr_quaternion.elements


def camera_axes_from_pyopenxr(
    xr_quaternion: Quaternionf,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pyopenxr quaternion to mujoco forward and up axes."""
    orientation = xr.Matrix4x4f.create_from_quaternion(xr_quaternion).as_numpy()
    forward = vector_from_pyopenxr(orientation[8:11])
    up = vector_from_pyopenxr(orientation[4:7])
    return forward, up
