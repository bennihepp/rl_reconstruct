from engine import BaseEngine


class DummyEngine(BaseEngine):

    class Exception(RuntimeError):
        pass

    def __init__(self, **kwargs):
        pass

    def close(self):
        raise NotImplementedError()

    def get_width(self):
        raise NotImplementedError()

    def get_height(self):
        raise NotImplementedError()

    def get_focal_length(self):
        raise NotImplementedError()

    def get_rgb_image(self):
        raise NotImplementedError()

    def get_rgb_image_by_file(self):
        raise NotImplementedError()

    def get_normal_rgb_image(self):
        raise NotImplementedError()

    def get_normal_rgb_image_by_file(self):
        raise NotImplementedError()

    def get_normal_image(self):
        raise NotImplementedError()

    def get_normal_image_by_file(self):
        raise NotImplementedError()

    def get_ray_distance_image(self):
        raise NotImplementedError()

    def get_ray_distance_image_by_file(self):
        raise NotImplementedError()

    def get_depth_image(self):
        raise NotImplementedError()

    def get_depth_image_by_file(self):
        raise NotImplementedError()

    def get_location(self):
        raise NotImplementedError()

    def get_orientation_rpy(self):
        raise NotImplementedError()

    def get_orientation_quat(self):
        raise NotImplementedError()

    def get_pose(self):
        raise NotImplementedError()

    def set_location(self, location):
        raise NotImplementedError()

    def set_orientation_rpy(self, roll, pitch, yaw):
        raise NotImplementedError()

    def set_orientation_quat(self, quat):
        raise NotImplementedError()

    def set_pose(self, pose):
        raise NotImplementedError()
