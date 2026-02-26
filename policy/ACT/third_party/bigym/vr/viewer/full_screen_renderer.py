"""Renders a full-screen image from NumPy array data using OpenGL."""
import inspect

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileShader, compileProgram

from vr.viewer import Side

VERTEX_SHADER = """
#version 150 core
out vec2 v_tex;

const vec2 pos[4]=vec2[4](vec2(-1.0, 1.0),
                          vec2(-1.0,-1.0),
                          vec2( 1.0, 1.0),
                          vec2( 1.0,-1.0));

void main()
{
    v_tex=0.5*pos[gl_VertexID] + vec2(0.5);
    gl_Position=vec4(pos[gl_VertexID], 0.0, 1.0);
}
"""
FRAGMENT_SHADER = """
#version 150 core
in vec2 v_tex;
uniform sampler2D texSampler;
out vec4 color;
void main()
{
    color=texture(texSampler, v_tex);
}
"""


class VRFullScreenRenderer:
    """Renders NumPy array to OpenGL texture."""

    def __init__(self, width: int, height: int):
        """Init.

        :param width (int): The width of the VR screen for one eye.
        :param height (int): The height of the VR screen for one eye.
        """
        self._width = width
        self._height = height
        self._shader = self._create_shader()
        self._vertex_array = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vertex_array)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0, 0, 0, 1)
        GL.glClearDepth(1.0)
        self._tex_ids = {
            Side.LEFT: self._create_texture(),
            Side.RIGHT: self._create_texture(),
        }

    @staticmethod
    def _create_shader() -> int:
        vertex_shader = compileShader(
            inspect.cleandoc(VERTEX_SHADER),
            GL.GL_VERTEX_SHADER,
        )
        fragment_shader = compileShader(
            inspect.cleandoc(FRAGMENT_SHADER),
            GL.GL_FRAGMENT_SHADER,
        )

        return compileProgram(vertex_shader, fragment_shader)

    @staticmethod
    def _create_texture() -> int:
        texid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texid)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texid

    @staticmethod
    def _update_texture(texture_id, width, height, data):
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGB,
            width,
            height,
            0,
            GL.GL_RGB,
            GL.GL_UNSIGNED_BYTE,
            data,
        )

    def render(self, side: Side, pixels: np.array):
        """Render pixels to the active buffer.

        Args:
            side: The side of the headset to render the pixels to.
            pixels: The pixel data to render.
        """
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # ToDo (nchernyadev): find better way to update textures
        #  instead of using slow glTexImage2D
        pixels = (
            pixels[:, : self._width, :]
            if side == Side.LEFT
            else pixels[:, self._width :, :]
        )
        self._update_texture(
            self._tex_ids[side],
            self._width,
            self._height,
            pixels,
        )

        # Render full-screen quad
        GL.glUseProgram(self._shader)
        GL.glBindVertexArray(self._vertex_array)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

        # Unbind texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
