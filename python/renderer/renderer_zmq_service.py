import numpy as np
import pvm
from pybh import serialization
from pybh import log_utils
from pybh import camera_utils
from pybh import math_utils
from pybh import utils


logger = log_utils.get_logger("RLrecon/render_server_zmq")


class RendererZMQService(object):

    def __init__(self, framebuffer, drawer, fov, initial_distance, trackball=None, world_to_opengl_mat=None):
        self._serializer = serialization.MsgPackSerializer()
        self._framebuffer = framebuffer
        self._drawer = drawer
        self._trackball = trackball
        if world_to_opengl_mat is None:
            world_to_opengl_mat = np.eye(4, dtype=np.float32)
        self._world_to_opengl_mat = world_to_opengl_mat
        self._opengl_to_world_mat = np.linalg.inv(self._world_to_opengl_mat)
        self._projection_transform = pvm.PerspectiveTransform(framebuffer.width, framebuffer.height, fov=fov)
        self._view_transform = pvm.ViewTransform()
        self._view_transform.z = - initial_distance
        self._input_enabled = True
        self._projection_transform_callbacks = utils.Callbacks()
        self._view_transform_callbacks = utils.Callbacks()
        self._window_visible_callbacks = utils.Callbacks()
        self._window_active_callbacks = utils.Callbacks()

    @property
    def input_enabled(self):
        return self._input_enabled

    @property
    def projection_transform(self):
        return self._projection_transform

    @property
    def view_transform(self):
        return self._view_transform

    @view_transform.setter
    def view_transform(self, view_transform):
        self._view_transform = view_transform

    @property
    def projection_transform_callback(self):
        return self._projection_transform_callbacks

    @property
    def view_transform_callback(self):
        return self._projection_transform_callbacks

    @property
    def window_visible_callbacks(self):
        return self._window_visible_callbacks

    @property
    def window_active_callbacks(self):
        return self._window_active_callbacks

    def handle_request_msg(self, request_dump, raise_exception=False):
        try:
            request = self._serializer.loads(request_dump)
            request_name = request.get(b"_request", None)
            response = None

            if request_name == b"ping":
                logger.info("Received 'ping' request")
                response = {b"ping": b"pong"}

            elif request_name == b"set":
                logger.debug("Received 'set' request")
                projection_transform_changed = False
                view_transform_changed = False
                for name, value in request.items():
                    if name.startswith(b"_"):
                        continue
                    if name == b"width":
                        # TODO
                        logger.warn("Changing of width and height is not implemented yet")
                        pass
                    elif name == b"height":
                        # TODO
                        logger.warn("Changing of width and height is not implemented yet")
                        pass
                    elif name == b"horz_fov":
                        horz_fov = value
                        focal_length = camera_utils.fov_to_focal_length(math_utils.degrees_to_radians(horz_fov), self._framebuffer.width)
                        vert_fov = math_utils.radians_to_degrees(camera_utils.focal_length_to_fov(focal_length, self._framebuffer.height))
                        logger.info("Setting horizontal FOV: {:.4f}, vertical FOV: {:.4f}".format(horz_fov, vert_fov))
                        self._projection_transform.fov = vert_fov
                        projection_transform_changed = True
                    elif name == b"location":
                        world_location = request[b"location"]
                        self._view_transform.location = - np.dot(self._world_to_opengl_mat[:3, :3], world_location)
                        view_transform_changed = True
                    elif name == b"orientation_rpy":
                        # TODO: There should also be a minus sign here
                        self._view_transform.orientation_rpy = request[b"orientation_rpy"]
                        view_transform_changed = True
                    elif name == b"input_enabled":
                        self._input_enabled = value
                    elif name == b"window_visible":
                        self._window_visible_callbacks(value)
                    elif name == b"window_active":
                        self._window_active_callbacks(value)
                    elif name == b"_request":
                        pass
                    else:
                        logger.error("Unknown field in 'get' request: {}".format(name))

                if projection_transform_changed:
                    self._projection_transform_callbacks(self._projection_transform)
                if view_transform_changed:
                    self._view_transform_callbacks(self._view_transform)

                response = {}

            elif request_name == b"get":
                logger.debug("Received 'get' request")
                response = {}
                for name in request.get(b"names", []):
                    if name == b"width":
                        response[b"width"] = self._framebuffer.width
                    elif name == b"height":
                        response[b"height"] = self._framebuffer.height
                    elif name == b"horz_fov":
                        vert_fov = self._projection_transform.fov
                        focal_length = camera_utils.fov_to_focal_length(math_utils.degrees_to_radians(vert_fov), self._framebuffer.height)
                        horz_fov = math_utils.radians_to_degrees(camera_utils.focal_length_to_fov(focal_length, self._framebuffer.width))
                        response[b"horz_fov"] = horz_fov
                    elif name == b"location":
                        # Minus sign because we move the world and not the camera
                        world_location = - np.dot(self._opengl_to_world_mat[:3, :3], self._view_transform.location)
                        response[b"location"] = world_location
                    elif name == b"orientation_rpy":
                        # TODO: There should also be a minus sign here
                        response[b"orientation_rpy"] = self._view_transform.orientation_rpy

            elif request_name == b"render_images":
                logger.debug("Received 'render_images' request")
                response = {}
                requested_images = [b"rgb_image", b"depth_image", b"normal_image"]
                projection_transform_changed = False
                view_transform_changed = False
                tmp_projection_transform = self._projection_transform
                tmp_view_transform = self._view_transform
                for name, value in request.items():
                    if name == b"location":
                        world_location = value
                        self._view_transform.location = - np.dot(self._world_to_opengl_mat[:3, :3], world_location)
                        view_transform_changed = True
                    elif name == b"orientation_rpy":
                        # TODO: There should also be a minus sign here
                        self._view_transform.orientation_rpy = value
                        view_transform_changed = True
                    elif name == b"use_trackball" and value:
                        if self._trackball is None:
                            logger.warn("Client asked for trackball viewpoint but no trackball was registered.")
                        else:
                            tmp_view_transform = self._trackball.transform
                    elif name == b"fov":
                        tmp_projection_transform = tmp_projection_transform.copy()
                        horz_fov = value
                        focal_length = camera_utils.fov_to_focal_length(math_utils.degrees_to_radians(horz_fov), self._framebuffer.width)
                        vert_fov = math_utils.radians_to_degrees(camera_utils.focal_length_to_fov(focal_length, self._framebuffer.height))
                        logger.info("Setting horizontal FOV: {:.4f}, vertical FOV: {:.4f}".format(horz_fov, vert_fov))
                        tmp_projection_transform.fov = vert_fov
                        projection_transform_changed = True
                    elif name == b"requested_images":
                        requested_images = value
                    elif name == b"_request":
                        pass
                    else:
                        logger.error("Unknown field in 'render_images' request: {}".format(name))

                if projection_transform_changed:
                    self._projection_transform_callbacks(self._projection_transform)
                if view_transform_changed:
                    self._view_transform_callbacks(self._view_transform)

                logger.debug("Requested images: {}".format(requested_images))
                with self._framebuffer.activate(clear=True):
                    self._drawer.draw(tmp_projection_transform.matrix, tmp_view_transform.matrix,
                                      self._framebuffer.width, self._framebuffer.height)
                if b"rgb_image" in requested_images:
                    rgb_pixels = self._framebuffer.read_rgba_pixels()
                    rgb_pixels = np.flip(rgb_pixels, 0)
                    response[b"rgb_image"] = rgb_pixels
                if b"depth_image" in requested_images:
                    depth_pixels = self._framebuffer.read_rgba_pixels(color_index=1)
                    depth_pixels = depth_pixels[:, :, 0]
                    depth_pixels = np.flip(depth_pixels, 0)
                    response[b"depth_image"] = depth_pixels
                if b"normal_image" in requested_images:
                    normal_pixels = self._framebuffer.read_rgba_pixels(color_index=2)
                    normal_pixels = normal_pixels[:, :, :3]
                    normal_pixels = np.flip(normal_pixels, 0)
                    response[b"normal_image"] = normal_pixels

            else:
                logger.warn("WARNING: Invalid request: {}".format(request))

            if response is not None:
                response_dump = self._serializer.dumps(response)
                # logger.debug("Sending response")
                return response_dump

        except Exception as exc:
            logger.exception("Exception when handling request")
            if raise_exception:
                raise
