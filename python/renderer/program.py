import contextlib
from glumpy import gloo, gl


class Program(object):

    def __init__(self, vertex_shader, fragment_shader, geometry_shader=None, **kwargs):
        self._program = gloo.Program(vertex_shader, fragment_shader, geometry_shader, **kwargs)

    @property
    def gl_handle(self):
        return self._program.handle

    @property
    def glumpy_program(self):
        return self._program

    def bind(self, data):
        self._program.bind(data)

    def draw(self, mode=gl.GL_TRIANGLES, indices=None):
        self._program.draw(mode, indices)

    @contextlib.contextmanager
    def activate(self):
        self._program.activate()
        yield
        self._program.deactivate()

    def deactivate(self):
        self._program.deactivate()

    def __getitem__(self, item):
        return self._program.__getitem__(item)

    def __setitem__(self, item, data):
        self._program.__setitem__(item, data)
