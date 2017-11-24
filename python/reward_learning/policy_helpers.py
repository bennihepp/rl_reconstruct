import numpy as np
from pybh import math_utils


class VisitedPoses(object):

    def __init__(self, vector_length, weights=None, tolerance=0.01):
        if weights is None:
            weights = np.ones((vector_length,))
        self._weights = np.array(weights)
        self._tolerance = tolerance
        self._visited_poses = np.zeros((0, vector_length))
        self._counts = np.empty((0,), dtype=np.int32)

    @property
    def counts(self):
        return self._counts

    def get_vector_from_pose(self, pose):
        location = pose.location()
        quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
        pose_vector = np.concatenate([location, quat])
        return pose_vector

    def increase_visited_pose(self, pose):
        if self._visited_poses.shape[0] > 0:
            idx, error = self.find_closest_pose(pose, return_error=True)
            print("inrease: pose={}, idx={}, error={}, counts={}".format(pose, idx, error, self._counts[idx]))
            if error < self._tolerance:
                self._counts[idx] += 1
                return
        pose_vector = self.get_vector_from_pose(pose)
        self._visited_poses = np.concatenate([self._visited_poses, pose_vector[np.newaxis, ...]])
        self._counts = np.append(self._counts, 1)

    def decrease_visited_pose(self, pose):
        if self._visited_poses.shape[0] > 0:
            idx, error = self.find_closest_pose(pose, return_error=True)
            print("decrease: pose={}, idx={}, error={}, counts={}".format(pose, idx, error, self._counts[idx]))
            if error < self._tolerance:
                assert(self._counts[idx] > 0)
                self._counts[idx] -= 1
            else:
                raise RuntimeError("Decrease visit count could not find any nearby visited pose (error={}).".format(error))
        else:
            raise RuntimeError("Decrease visit count should only be called on already visited poses.")

    def get_distances(self, pose):
        pose_vector = self.get_vector_from_pose(pose)
        error = pose_vector - self._visited_poses
        distances = np.square(np.sum(np.square(self._weights * error), axis=-1))
        return distances

    def find_closest_pose(self, pose, return_error=False, weights=None):
        if weights is None:
            weights = self._weights
        assert (self._visited_poses.shape[0] > 0)
        pose_vector = self.get_vector_from_pose(pose)
        error = pose_vector - self._visited_poses
        # TODO: This was used for dataset generation but is not really what was intended.
        distances_sq = np.square(np.sum(np.square(weights * error), axis=-1))
        idx = np.argmin(distances_sq, axis=0)
        if return_error:
            return idx, distances_sq[idx]
        else:
            return idx

    def find_k_closest_poses(self, pose, k, return_error=False, weights=None):
        if weights is None:
            weights = self._weights
        if self._visited_poses.shape[0] == 0:
            if return_error:
                return [], []
            else:
                return []
        assert (self._visited_poses.shape[0] > 0)
        pose_vector = self.get_vector_from_pose(pose)
        error = pose_vector - self._visited_poses
        distances_sq = np.sum(np.square(weights * error), axis=-1)
        distances_sorted_indices = np.argsort(distances_sq)
        k_indices = distances_sorted_indices[:k]
        if return_error:
            k_distances = distances_sq[k_indices]
            return k_indices, k_distances
        else:
            return k_indices

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
