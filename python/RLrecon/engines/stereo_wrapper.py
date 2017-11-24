from __future__ import print_function
import numpy as np
import cv2
from pybh.contrib import transformations
from pybh import math_utils
from pybh import log_utils


logger = log_utils.get_logger("RLrecon/engines/engine")


class StereoWrapper(object):

    def __init__(self,
                 base_engine,
                 stereo_method,
                 stereo_baseline,
                 width,
                 height,
                 min_depth,
                 num_disparities=64,
                 block_size=None):
        self._base_engine = base_engine
        self._stereo_method = stereo_method
        self._stereo_baseline = stereo_baseline
        self._width = width
        self._height = height
        self._min_depth = min_depth
        self._num_disparities = num_disparities
        self._focal_length = base_engine.get_focal_length()
        self._left_matcher = self._create_left_matcher(stereo_method, num_disparities, block_size)
        self._right_matcher = self._create_right_matcher(self._left_matcher)
        self._wls_filter = self._create_wls_filter(self._left_matcher)
        print("Min depth={}".format(stereo_baseline * self._focal_length / float(num_disparities)))
        print("Max depth={}".format(stereo_baseline * self._focal_length / float(1)))

    def _create_left_matcher(self, stereo_method, num_disparities, block_size):
        if stereo_method == "bm":
            if block_size is None or block_size <= 0:
                block_size = 21
            matcher = cv2.StereoBM_create(num_disparities, block_size)
        elif stereo_method == "sgbm":
            if block_size is None or block_size <= 0:
                block_size = 5
            matcher = cv2.StereoSGBM_create(0, num_disparities, block_size)
            matcher.setP1(24 * block_size * block_size)
            matcher.setP2(48 * block_size * block_size)
            matcher.setMode(cv2.StereoSGBM_MODE_SGBM_3WAY)
        else:
            raise ValueError("Unknown stereo method: {}".format(stereo_method))
        return matcher

    def _create_right_matcher(self, left_matcher):
        return cv2.ximgproc.createRightMatcher(left_matcher)

    def _create_wls_filter(self, left_matcher, lambda_=8000.0, sigma=1.5):
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(lambda_)
        wls_filter.setSigmaColor(sigma)
        return wls_filter

    def _compute_disparity_image(self, left_img, right_img):
        left_disp = self._left_matcher.compute(left_img, right_img)
        right_disp = self._right_matcher.compute(right_img, left_img)
        print(left_disp.shape)
        print(left_img.shape)
        print(right_disp.shape)
        filtered_disp = self._wls_filter.filter(left_disp, left_img, None, right_disp)
        # conf_map = self._wls_filter.getConfidenceMap()
        # roi = self._wls_filter.getROI()
        filtered_disp = filtered_disp.astype(np.float32) / 16.0
        return filtered_disp

    def _compute_depth_image(self, disp_img):
        depth_img = self._stereo_baseline * self._focal_length / disp_img
        depth_img[:, :self._num_disparities] = 0
        depth_img[disp_img <= 0] = np.finfo(np.float32).max
        depth_img[depth_img <= self._min_depth] = 0.0
        depth_img[:, :self._num_disparities] = 0.0
        return depth_img

    def get_depth_image(self):
        location, orientation_rpy = self._base_engine.get_pose_rpy()
        quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        stereo_offset = np.array([0, - self._stereo_baseline, 0])
        stereo_offset = math_utils.rotate_vector_with_quaternion(quat, stereo_offset)
        # print(stereo_offset)
        pose1 = (location, orientation_rpy)
        pose2 = (location + stereo_offset, orientation_rpy)
        self._base_engine.set_pose_rpy(pose1, wait_until_set=True)
        rgb_image1 = self._base_engine.get_rgb_image()
        rgb_image1 = cv2.cvtColor(rgb_image1, cv2.COLOR_RGB2GRAY)
        rgb_image1 = (255 * rgb_image1).astype(np.uint8)
        self._base_engine.set_pose_rpy(pose2, wait_until_set=True)
        rgb_image2 = self._base_engine.get_rgb_image()
        # rgb_image2 = environment.base.get_engine().get_normal_image()
        rgb_image2 = (255 * rgb_image2).astype(np.uint8)
        rgb_image2 = cv2.cvtColor(rgb_image2, cv2.COLOR_RGB2GRAY)

        disp_img = self._compute_disparity_image(rgb_image1, rgb_image2)
        print("Minimum disparity: {}".format(np.min(disp_img)))
        print("Maximum disparity: {}".format(np.max(disp_img)))
        # disp_img[:50, :] = 0.5
        depth_img = self._compute_depth_image(disp_img)

        # disparity_img = self._stereo_matcher.compute(rgb_image1, rgb_image2)
        # # OpenCV disparity image is represented as a fixed point disparity map with 4 fractional bits
        # disparity_img = disparity_img.astype(np.float32) / 16.0
        # print("Minimum disparity: {}".format(np.min(disparity_img)))
        # print("Maximum disparity: {}".format(np.max(disparity_img)))
        # # print("focal_length: {}".format(self._focal_length))
        # depth_img = self._stereo_baseline * self._focal_length / disparity_img
        # depth_img[:, :self._num_disparities] = 0
        # depth_img[disparity_img <= 0] = np.finfo(np.float32).max
        # # depth_img[disparity_img == 0] = 10.0
        self._base_engine.set_pose_rpy(pose1, wait_until_set=True)
        gt_depth_img = self._base_engine.get_depth_image()
        # depth_img[depth_img <= self._min_depth] = -1
        # depth_img[:, :self._num_disparities] = -1\

        # # Overwrite depth for far away pixels (problem with sky and block matching)
        # mask = gt_depth_img > 100
        # depth_img[mask] = gt_depth_img[mask]
        # depth_img[:, :self._num_disparities] = 0

        # depth_img[depth_img > 10] = 10.0
        # gt_depth_img[gt_depth_img > 10] = 10.0
        # error_img = np.abs(depth_img - gt_depth_img)
        # print(np.max(error_img))

        depth_img_copy = depth_img.copy()
        depth_img_copy[depth_img_copy > 10] = 10.0
        gt_depth_img_copy = gt_depth_img.copy()
        gt_depth_img_copy[gt_depth_img_copy > 10] = 10.0
        cv2.imwrite("disp_img_{}.png".format(self._stereo_method), disp_img.astype(np.uint8))
        cv2.imwrite("depth_img_{}.png".format(self._stereo_method), 255 * depth_img_copy / 10.)
        cv2.imwrite("gt_depth_img_{}.png".format(self._stereo_method), 255 * gt_depth_img_copy / 10.)
        # cv2.imwrite("gt_depth_img.png", 255 * gt_depth_img / 10.)
        # cv2.imwrite("error_img.png", 255 * error_img / 10.)
        cv2.imwrite("rgb_image1_{}.png".format(self._stereo_method), rgb_image1)
        cv2.imwrite("rgb_image2_{}.png".format(self._stereo_method), rgb_image2)
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.hist(disparity_img.flatten())
        # plt.figure()
        # plt.hist(depth_img.flatten())
        # plt.show()
        # return None
        # dsize = (64, 64)
        dsize = (self._width, self._height)
        # depth_img = gt_depth_img
        depth_img = cv2.resize(depth_img, dsize=dsize, interpolation=cv2.INTER_NEAREST)
        return depth_img
        # cv2.imshow("rgb_image1", rgb_image1)
        # cv2.imshow("rgb_image2", rgb_image2)
        # cv2.imshow("disparity", disparity_img)
        # cv2.imshow("depth", depth_img)
        # cv2.imshow("gt_depth", gt_depth_img)
        # print(np.min(disparity_img), np.max(disparity_img))
        # print(np.min(depth_img), np.max(depth_img))
        # cv2.waitKey()
        # return depth_img

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_focal_length(self):
        scale_factor = self.get_width() / float(self._base_engine.get_width())
        return self._base_engine.get_focal_length() * scale_factor

    def get_intrinsics(self):
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = self.get_focal_length()
        intrinsics[1, 1] = self.get_focal_length()
        intrinsics[0, 2] = (self.get_width() - 1) / 2.0
        intrinsics[1, 2] = (self.get_height() - 1) / 2.0
        intrinsics[2, 2] = 1.0
        return intrinsics

    def get_horizontal_field_of_view_degrees(self):
        return self._base_engine.get_horizontal_field_of_view_degrees()

    def __getattr__(self, attr):
        print("Returning {} attribute from base engine".format(attr))
        return getattr(self._base_engine, attr)
