import contextlib
import numpy as np
from glumpy import gloo, gl
import opengl_utils


class Framebuffer(object):

    def __init__(self, width, height, num_color_attachments=1, color_dtypes=None, has_depth=True):
        assert num_color_attachments >= 1, "Need at least one color attachment"
        if color_dtypes is None:
            color_dtypes = [np.ubyte] * num_color_attachments
        assert len(color_dtypes) == num_color_attachments
        self._color_attachments = []
        for i in range(num_color_attachments):
            color = np.zeros((height, width, 4), color_dtypes[i])
            if color_dtypes[i] == np.float32:
                color = color.view(gloo.TextureFloat2D)
            else:
                color = color.view(gloo.Texture2D)
            color.interpolation = gl.GL_LINEAR
            self._color_attachments.append(color)
        self._depth_buffer = gloo.DepthBuffer(width, height)
        self._framebuffer = gloo.FrameBuffer(color=self._color_attachments, depth=self._depth_buffer)

    def clear(self):
        pass

    @property
    def color_attachment(self):
        return self._color_attachments[0]

    @property
    def color_attachments(self):
        return self._color_attachments

    @property
    def depth_buffer(self):
        return self._depth_buffer

    @property
    def native_framebuffer(self):
        return self._framebuffer

    @property
    def width(self):
        return self._framebuffer.width

    @property
    def height(self):
        return self._framebuffer.height

    @contextlib.contextmanager
    def activate(self, clear=False):
        self._framebuffer.activate()
        gl.glViewport(0, 0, self.width, self.height)
        if clear:
            # color_attachment_flags = []
            # for i in range(len(self._color_attachments)):
            #     color_attachment_flags.append(gl.GL_COLOR_ATTACHMENT0 + i)
            # gl.glDrawBuffers(np.array(color_attachment_flags, dtype=np.uint32))
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        yield self
        self._framebuffer.deactivate()

    def read_rgba_pixels(self, color_index=0, activate=True):
        if activate:
            with self.activate():
                return self.read_rgba_pixels(color_index, activate=False)

        np_type = self._color_attachments[color_index].dtype
        gl_type = opengl_utils.get_gl_type(np_type)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + color_index)
        pixels = gl.glReadPixels(0, 0, self._framebuffer.width, self._framebuffer.height,
                                 gl.GL_RGBA, gl_type, outputType=np_type)
        # Bug in OpenGL binding. OpenGL is stored in row-major order (row being a horizontal line of an image).
        pixels = pixels.reshape((self.height, self.width, -1))
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        return pixels

    def read_depth_pixels(self, activate=True):
        if activate:
            with self.activate():
                return self.read_depth_pixels(activate=False)
        np_type = self._depth_buffer.dtype
        gl_type = opengl_utils.get_gl_type(np_type)
        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, outputType=np_type)
        # Bug in OpenGL binding. OpenGL is stored in row-major order (row being a horizontal line of an image).
        pixels = pixels.reshape((self.height, self.width, -1))
        return pixels

    def draw_rgba_pixels(self, pixels, color_index=0, activate=True):
        if activate:
            with self.activate():
                return self.draw_rgba_pixels(pixels, color_index, activate=False)
        gl_type = opengl_utils.get_gl_type(pixels.dtype)
        gl.glDrawBuffers(np.array([gl.GL_COLOR_ATTACHMENT0 + color_index], dtype=np.uint32))
        gl.glDrawPixels(self.width, self.height, gl.GL_RGBA, gl_type, pixels)

    def draw_depth_pixels(self, pixels, activate=True):
        assert pixels.dtype == np.float32
        if activate:
            with self.activate():
                return self.draw_depth_pixels(pixels, activate=False)
        gl.glDrawPixels(self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, pixels)
