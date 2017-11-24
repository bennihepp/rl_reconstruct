import glumpy


class Application(object):

    def __init__(self, backend_name=None):
        if backend_name is not None:
            glumpy.app.use(backend_name)
        self._backend = glumpy.app.__backend__
        # assert self._window_count > 0, "A window has to be created before Application can be initialized"
        self._clock = glumpy.app.__init__(backend=self._backend)

    def get_num_windows(self):
        if self._backend.windows() is None:
            return 0
        else:
            return len(self._backend.windows())

    def process(self, dt=None):
        if dt is None:
            dt = self._clock.tick()
        self._backend.process(dt)

    @property
    def clock(self):
        return self._clock
