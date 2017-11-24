import math_utils
import ros_utils
import rospy
from RLrecon.environments.environment import SimpleV1Environment
from engines.unreal_cv_wrapper_old import UnrealCVWrapper


def run(environment):
    environment.reset(reset_map=False)

    # Setup loop timer
    rate = rospy.Rate(10)

    # level = 0
    # size_x = 4
    # level = 3
    # size_x = 16
    level = 5
    size_x = 4
    size_y = size_x
    size_z = size_x

    while not rospy.is_shutdown():
        pose = environment.get_pose()
        center = pose.location()
        orientation_rpy = pose.orientation_rpy()
        # orientation_rpy[0] = 0
        # orientation_rpy[1] = 0
        res = environment._mapper.perform_query_subvolume_rpy(
            center, orientation_rpy, level, size_x, size_y, size_z)
        print("Query subvolume took {}s".format(res.elapsed_seconds))
        print("Size of queried subvolume: {}".format(len(res.occupancies)))
        print("Size of voxel in queried subvolume: {}".format(res.voxel_size))
        print("Min voxel: {}, max voxel: {}".format(ros_utils.point_ros_to_numpy(res.voxel_min),
                                                    ros_utils.point_ros_to_numpy(res.voxel_max)))

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('test_agent', anonymous=False)
    bounding_box = math_utils.BoundingBox(
        [-10, -10, 0],
        [+10, +10, +10]
    )
    score_bounding_box = math_utils.BoundingBox(
        [-10, -10, -10],
        [+10, +10, +10]
    )
    engine = UnrealCVWrapper(max_depth_viewing_angle=math_utils.degrees_to_radians(70.))
    # environment = Environment(bounding_box)
    environment = SimpleV1Environment(
        bounding_box, score_bounding_box=score_bounding_box, engine=engine, clear_size=6.0)
    run(environment)
