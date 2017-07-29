import sys
import struct
import numpy as np
import cv2
import rospy
import tf
from tf import transformations
import tf_conversions
from geometry_msgs.msg import Transform
from octomap_server_ext.srv import InsertPointCloud, InsertPointCloudRequest
from RLrecon.engine.unreal_cv_wrapper import UnrealCVWrapper


def run(engine):
    # Setup loop timer
    rate = rospy.Rate(10)

    # Initial yaw angle mode (0 -> 0, 1 -> np.pi)
    mode = 0

    while not rospy.is_shutdown():
        # Perform action:
        # Update camera orientation
        if mode == 0:
            location = np.array([-5, 0, 5])
            yaw = 0
        elif mode == 1:
            location = np.array([-5, 0, 5])
            yaw = np.pi
        elif mode == 2:
            location = np.array([0, 5, 5])
            yaw = np.pi / 2
        else:
            location = np.array([0, 5, 5])
            yaw = np.pi * 3 / 2
        roll = 0
        pitch = 0
        # if yaw >= 2 * np.pi:
        #     yaw -= 2 * np.pi
        # elif yaw < 0:
        #     yaw += 2 * np.pi
        engine.set_orientation_rpy(roll, pitch, yaw)
        # Update camera location
        engine.set_location(location)
        # rospy.logdebug("location: {} {} {}".format(*new_location))
        # rospy.logdebug("rotation: {} {} {}".format(roll * 180 / np.pi, pitch * 180 / np.pi, yaw * 180 / np.pi))

        # Read new pose, camera info and depth image
        depth_image = engine.get_depth_image()
        rospy.loginfo("mode={}, min depth={}, max depth={}".format(
            mode, np.min(depth_image.flatten()), np.max(depth_image.flatten())))
        if mode == 0:
            assert(np.min(depth_image) < 4)
        elif mode == 1:
            assert(np.min(depth_image) > 6)
        elif mode == 2:
            assert(np.min(depth_image) > 6)
        else:
            assert(np.min(depth_image) < 4)
        # Update camera state
        mode += 1
        mode = mode % 4

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('test_depth_maps', anonymous=False)
    engine = UnrealCVWrapper(image_scale_factor=0.5)
    run(engine)
