import numpy as np
from RLrecon import math_utils


class VisitedPoses(object):

    def __init__(self, vector_length, weights=None, tolerance=0.01):
        if weights is None:
            weights = np.ones((vector_length,))
        else:
            self._weights = np.array(weights)
        self._tolerance = tolerance
        self._visited_poses = np.zeros((0, vector_length))
        self._counts = []

    def get_vector_from_pose(self, pose):
        location = pose.location()
        quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
        pose_vector = np.concatenate([location, quat])
        return pose_vector

    def add_visited_pose(self, pose):
        if self._visited_poses.shape[0] > 0:
            idx, error = self.find_closest_pose(pose, return_error=True)
            if error < self._tolerance:
                self._counts[idx] += 1
                return
        pose_vector = self.get_vector_from_pose(pose)
        self._visited_poses = np.concatenate([self._visited_poses, pose_vector[np.newaxis, ...]])
        self._counts.append(1)

    def get_distances(self, pose):
        pose_vector = self.get_vector_from_pose(pose)
        error = pose_vector - self._visited_poses
        distances = np.square(np.sum(np.square(self._weights * error), axis=-1))
        return distances

    def find_closest_pose(self, pose, return_error=False):
        assert (self._visited_poses.shape[0] > 0)
        pose_vector = self.get_vector_from_pose(pose)
        error = pose_vector - self._visited_poses
        distances = np.square(np.sum(np.square(self._weights * error), axis=-1))
        idx = np.argmin(distances, axis=0)
        if return_error:
            return idx, distances[idx]
        else:
            return idx

    def has_been_visited_before(self, pose):
        distances = self.get_distances(pose)
        if np.any(distances < self._tolerance):
            return True
        return False

    def get_visit_count(self, pose):
        if self._visited_poses.shape[0] == 0:
            return False
        idx, error = self.find_closest_pose(pose, return_error=True)
        if error < self._tolerance:
            return self._counts[idx]
        else:
            return 0
