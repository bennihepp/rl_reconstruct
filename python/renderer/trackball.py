# Adapted from glumpy to make it suitable for shader integration without pattern replacement.
#
# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------

import numpy as np
from glumpy import glm
import pvm


class Trackball(object):
    """
    3D trackball transform

    :param float aspect:
       Indicate what is the aspect ratio of the object displayed. This is
       necessary to convert pixel drag move in oject space coordinates.
       Default is None.

    :param float znear:
       Near clip plane. Default is 2.

    :param float zfar:
       Distance clip plane. Default is 1000.

    :param float theta:
       Angle (in degrees) around the z axis. Default is 45.

    :param float phi:
       Angle (in degrees) around the x axis. Default is 45.

    :param float distance:
       Distance from the trackball to the object.  Default is 8.

    :param float zoom:
           Zoom level. Default is 35.

    The trackball transform simulates a virtual trackball (3D) that can rotate
    around the origin using intuitive mouse gestures.

    The transform is connected to the following events:

      * ``on_attach``: Transform initialization
      * ``on_resize``: Tranform update to maintain aspect
      * ``on_mouse_scroll``: Zoom in & out (user action)
      * ``on_mouse_grab``: Drag (user action)

    **Usage example**:

      .. code:: python

         vertex = '''
         attribute vec2 position;
         void main()
         {
             gl_Position = <transform>(vec4(position, 0.0, 1.0));
         } '''

         ...
         window = app.Window(width=800, height=800)
         program = gloo.Program(vertex, fragment, count=4)
         ...
         program['transform'] = Trackball(aspect=1)
         window.attach(program['transform'])
         ...
    """

    def __init__(self, pitch=0, yaw=0, z_offset=10, aspect=1):
        """
        Initialize the transform.
        """
        self._aspect = aspect
        self._pitch = pitch
        self._yaw = yaw
        self._roll = 0
        self._z_offset = z_offset
        self._width = 1
        self._height = 1
        self._x_offset = 0
        self._y_offset = 0

        self._transform = pvm.ModelTransform()
        self._update_view()

    @property
    def distance(self):
        """ Distance from the trackball to the object """
        return self._distance

    @distance.setter
    def distance(self, distance):
        """ Distance from the trackball to the object """
        self._distance = abs(distance)
        self._update_view()

    @property
    def yaw(self):
        """ Angle (in degrees) around the yaw axis """
        return self._yaw

    @yaw.setter
    def yaw(self, yaw):
        """ Angle (in degrees) around the yaw axis """
        self._yaw = yaw
        self._update_view()

    @property
    def pitch(self):
        """ Angle (in degrees) around the pitch axis """
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        """ Angle (in degrees) around the pitch axis """
        self._pitch = pitch
        self._update_view()

    @property
    def camera_location(self):
        print(self._transform.matrix.T)
        origin_vec = np.array([0.0, 0.0, 0.0, 1.0])
        # print(origin_vec)
        # print(np.dot(self._transform.matrix.T, origin_vec))
        # cam_location = np.dot(self._transform.matrix.T, origin_vec)
        cam_location = np.dot(self._transform.inv_matrix.T, origin_vec)
        cam_location /= cam_location[-1]
        # cam_location = self.location
        return cam_location[:3]

    @property
    def location(self):
        return np.array([self._x_offset, self._y_offset, -self._z_offset])

    @property
    def orientation_rpy(self):
        return np.array([self._roll, self._pitch, self._yaw])

    def _update_view(self):
        location = np.array([self._x_offset, self._y_offset, -self._z_offset])
        self._transform.location = location
        orientation_rpy = np.array([self._roll, self._pitch, self._yaw])
        self._transform.orientation_rpy = orientation_rpy

    @property
    def view(self):
        # return self._view
        return self._transform.matrix

    @property
    def transform(self):
        return self._transform

    def on_resize(self, width, height):
        self._width = float(width)
        self._height = float(height)

    def on_mouse_drag(self, x, y, dx, dy, button):
        width = self._width
        height = self._height
        # x = (x*2.0 - width)/width
        dx = dx / width
        # y = (height - y*2.0)/height
        dy = -dy / height
        if button == 2:
            self._yaw -= 50 * dx
            self._pitch -= 50 * dy
        elif button == 8:
            self._x_offset += 5 * dx
            self._y_offset += 5 * dy
        self._update_view()

    def on_mouse_scroll(self, x, y, dx, dy):
        self._z_offset += dy
        self._update_view()
