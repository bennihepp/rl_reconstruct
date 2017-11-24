import contextlib
import numpy as np
import glumpy
from glumpy import gl, glm
import opengl_utils


class Window(object):

    def __init__(self, fov=45.0, znear=0.5, zfar=10000, do_clear=True, visible=True, **kwargs):
        self._fov = fov
        self._znear = znear
        self._zfar = zfar
        self._do_clear = do_clear
        self._drawers = []
        self._win = glumpy.app.Window(visible=visible, **kwargs)
        self._win.push_handlers(on_resize=self._on_resize,
                                on_init=self._on_init,
                                on_draw=self._on_draw)
        self._projection = None
        self._view = None
        self._width = 1
        self._height = 1
        self._visible = visible

    def show(self):
        self._visible = True
        self._win.show()

    def hide(self):
        self._visible = False
        self._win.hide()

    @property
    def visible(self):
        # Note: This could be wrong if the glumpy window object was modified directly
        return self._visible

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        self._fov = fov

    @property
    def znear(self):
        return self._znear

    @znear.setter
    def znear(self, znear):
        self._znear = znear

    @property
    def zfar(self):
        return self._zfar

    @zfar.setter
    def zfar(self, zfar):
        self._zfar = zfar

    @property
    def do_clear(self):
        return self._do_clear

    @do_clear.setter
    def do_clear(self, do_clear):
        self._do_clear = do_clear

    def add_drawer(self, drawer):
        if callable(drawer):
            self._drawers.append(drawer)
        else:
            self._drawers.append(drawer.draw)

    def clear_drawers(self):
        self._drawers = []

    def remove_drawer(self, drawer):
        self._drawers.remove(drawer)

    def _on_init(self):
        pass

    def _on_resize(self, width, height):
        aspect = width / float(height)
        self._projection = glm.perspective(self._fov, aspect, self._znear, self._zfar)
        self._view = np.eye(4, dtype=np.float32)
        self._width = width
        self._height = height

    def _on_draw(self, dt):
        self._win.clear()
        with self.activate(self._do_clear):
            for drawer in self._drawers:
                drawer(self._projection, self._view, self._width, self._height)

    @property
    def projection(self):
        return self._projection

    @property
    def view(self):
        return self._view

    def override_projection(self, projection):
        self._projection = projection

    def override_view(self, view):
        self._view = view.copy()

    @property
    def glumpy_window(self):
        return self._win

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @contextlib.contextmanager
    def activate(self, clear=False):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self.width, self.height)
        if clear:
            # self._win.clear()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        yield self

    def read_rgba_pixels(self, activate=True, dtype=np.ubyte):
        if activate:
            with self.activate():
                return self.read_rgba_pixels(activate=False)

        gl_type = opengl_utils.get_gl_type(dtype)
        pixels = gl.glReadPixels(0, 0, self.width, self.height,
                                 gl.GL_RGBA, gl_type, outputType=dtype)
        pixels = pixels.reshape((self.height, self.width, -1))
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        return pixels

    def read_depth_pixels(self, activate=True):
        if activate:
            with self.activate():
                return self.read_depth_pixels(activate=False)
        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, outputType=np.float32)
        pixels = pixels.reshape((self.height, self.width, -1))
        return pixels

    def draw_rgba_pixels(self, pixels, activate=True):
        if activate:
            with self.activate():
                return self.draw_rgba_pixels(pixels, activate=False)
        gl_type = opengl_utils.get_gl_type(pixels.dtype)
        gl.glDrawPixels(self.width, self.height, gl.GL_RGBA, gl_type, pixels)

    def draw_depth_pixels(self, pixels, activate=True):
        assert pixels.dtype == np.float32
        if activate:
            with self.activate():
                return self.draw_depth_pixels(pixels, activate=False)
        gl.glDrawPixels(self.width, self.height, gl.GL_RGBA, gl.GL_FLOAT, pixels)
