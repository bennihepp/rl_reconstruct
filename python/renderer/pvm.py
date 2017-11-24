import numpy as np
from glumpy import glm


class PerspectiveTransform(object):

    def __init__(self, width, height, fov=45.0, znear=0.5, zfar=10000):
        self._width = width
        self._height = height
        self._fov = fov
        self._znear = znear
        self._zfar = zfar
        self._update_matrix()

    def copy(self):
        return PerspectiveTransform(self.width, self.height, self.fov, self.znear, self.zfar)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width
        self._update_matrix()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = height
        self._update_matrix()

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov):
        self._fov = fov
        self._update_matrix()

    @property
    def znear(self):
        return self._znear

    @znear.setter
    def znear(self, znear):
        self._znear = znear
        self._update_matrix()

    @property
    def zfar(self):
        return self._zfar

    @zfar.setter
    def zfar(self, zfar):
        self._zfar = zfar
        self._update_matrix()

    def _update_matrix(self):
        aspect = self._width / float(self._height)
        self._matrix = glm.perspective(self._fov, aspect, self._znear, self._zfar)

    def on_resize(self, width, height):
        self._width = width
        self._height = height
        self._upate_matrix()

    @property
    def matrix(self, copy=True):
        if copy:
            return np.array(self._matrix)
        else:
            return self._matrix


class _RigidTransform(object):

    def __init__(self, location=None, orientation_rpy=None, scale=None):
        if location is None:
            location = np.array([0, 0, 0], dtype=np.float32)
        self._location = location
        if orientation_rpy is None:
            self._yaw = 0
            self._pitch = 0
            self._roll = 0
        else:
            self._yaw = orientation_rpy[2]
            self._pitch = orientation_rpy[1]
            self._roll = orientation_rpy[0]
        if scale is None:
            scale = np.array([1, 1, 1], dtype=np.float32)
        self._scale = scale
        self._update_matrix()

    def copy(self):
        raise NotImplementedError()
        return _RigidTransform(self.location, self.orientation_rpy, self.scale)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale[:] = scale
        self._update_matrix()

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, location):
        self._location = np.array(location)
        self._update_matrix()

    @property
    def x(self):
        return self._location[0]

    @x.setter
    def x(self, x):
        self._location[0] = x
        self._update_matrix()

    @property
    def y(self):
        return self._location[1]

    @x.setter
    def y(self, y):
        self._location[1] = y
        self._update_matrix()

    @property
    def z(self):
        return self._location[2]

    @z.setter
    def z(self, z):
        self._location[2] = z
        self._update_matrix()

    @property
    def orientation_rpy(self):
        return np.array([self._roll, self._pitch, self._yaw])

    @orientation_rpy.setter
    def orientation_rpy(self, orientation_rpy):
        self._yaw = orientation_rpy[2]
        self._pitch = orientation_rpy[1]
        self._roll = orientation_rpy[0]
        self._update_matrix()

    @property
    def yaw(self):
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        self._yaw = yaw
        self._update_matrix()

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._pitch = pitch
        self._update_matrix()

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, roll):
        self._roll = roll
        self._update_matrix()

    def _update_matrix(self):
        raise NotImplementedError()
        self._initialize_to_identity()
        self._apply_scale(self._matrix)
        self._apply_rotation(self._matrix)
        self._apply_translation(self._matrix)
        self._update_inv_matrix()

    def _update_inv_matrix(self):
        # TODO: This could be done without matrix inversion
        self._inv_matrix = np.linalg.inv(self._matrix)

    @property
    def matrix(self, copy=True):
        if copy:
            return np.array(self._matrix)
        else:
            return self._matrix

    @property
    def inv_matrix(self, copy=True):
        if copy:
            return np.array(self._inv_matrix)
        else:
            return self._inv_matrix

    def _initialize_to_identity(self):
        self._matrix = np.eye(4, dtype=np.float32)

    def _apply_scale(self):
        glm.scale(self._matrix, *self._scale)

    def _apply_translation(self):
        glm.translate(self._matrix, *self._location)

    def _apply_rotation(self):
        glm.yrotate(self._matrix, self._yaw)
        glm.xrotate(self._matrix, -self._pitch)
        glm.zrotate(self._matrix, -self._roll)


class ViewTransform(_RigidTransform):

    """The view transform first translates the camera and then rotates the world around it"""

    def copy(self):
        return ViewTransform(self.location, self.orientation_rpy, self.scale)

    def _update_matrix(self):
        self._initialize_to_identity()
        self._apply_scale()
        self._apply_translation()
        self._apply_rotation()
        self._update_inv_matrix()


class ModelTransform(_RigidTransform):

    """The model transform first rotates the model and the translates it"""

    def copy(self):
        return ModelTransform(self.location, self.orientation_rpy, self.scale)

    def _update_matrix(self):
        self._initialize_to_identity()
        self._apply_scale()
        self._apply_rotation()
        self._apply_translation()
        self._update_inv_matrix()
