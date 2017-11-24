import numpy as np
from glumpy import gl
from program import Program
import opengl_utils


class FramebufferDrawer(object):

    DRAW_PIXELS = 1
    DRAW_QUAD = 2
    BLIT = 3

    VERTEX = """
    attribute vec2 a_position;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texcoord = (a_position + 1.0) / 2.0;
    }
    """

    FRAGMENT = """
    uniform float u_color_scale;
    uniform float u_color_offset;
    uniform sampler2D u_texture;
    varying vec2 v_texcoord;
    void main()
    {
        vec4 color = texture2D(u_texture, v_texcoord);
        color = vec4(u_color_offset + u_color_scale * color.xyz, color.w);
        gl_FragColor = clamp(color, 0, 1);
    }
    """

    def __init__(self, framebuffer, color_index=0, draw_mode=DRAW_PIXELS):
        self._draw_mode = draw_mode
        self._framebuffer = framebuffer
        self._color_offset = 0.0
        self._color_scale = 1.0
        if self._draw_mode == self.DRAW_PIXELS:
            self._color_index = color_index
        elif self._draw_mode == self.DRAW_QUAD:
            self._quad = Program(self.VERTEX, self.FRAGMENT, count=4)
            self._quad["a_position"] = (-1, -1), (-1, +1), (+1, -1), (+1, +1)
            self._quad["u_texture"] = self._framebuffer.color_attachments[color_index]
        elif self._draw_mode == self.BLIT:
            self._color_index = color_index
        else:
            raise RuntimeError("Unknown drawing mode: {}".format(self._draw_mode))

    def set_color_offset(self, offset):
        assert self._draw_mode == self.DRAW_PIXELS or self._draw_mode == self.DRAW_QUAD
        self._color_offset = offset

    def set_color_scale(self, scale):
        assert self._draw_mode == self.DRAW_PIXELS or self._draw_mode == self.DRAW_QUAD
        self._color_scale = scale

    def draw(self, projection, view, width, height):
        if self._draw_mode == self.DRAW_PIXELS:
            np_type = self._framebuffer.color_attachments[self._color_index].dtype
            gl_type = opengl_utils.get_gl_type(np_type)
            pixels = self._framebuffer.read_rgba_pixels(self._color_index)
            if self._color_scale != 1.0:
                pixels *= self._color_scale
            if self._color_offset != 1.0:
                pixels += self._color_offset
            width = min(width, self._framebuffer.width)
            height = min(height, self._framebuffer.height)
            gl.glDrawPixels(width, height, gl.GL_RGBA, gl_type, pixels)
        elif self._draw_mode == self.DRAW_QUAD:
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            with self._quad.activate():
                self._quad["u_color_offset"] = self._color_offset
                self._quad["u_color_scale"] = self._color_scale
                self._quad.draw(gl.GL_TRIANGLE_STRIP)
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glEnable(gl.GL_DEPTH_TEST)
        elif self._draw_mode == self.BLIT:
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._framebuffer.native_framebuffer.handle)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + self._color_index)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
            gl.glBlitFramebuffer(0, 0, self._framebuffer.width, self._framebuffer.height,
                                 0, 0, width, height,
                                 gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        else:
            raise RuntimeError("Unknown drawing mode")
