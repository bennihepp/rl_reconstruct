import math_utils
import rospy
from RLrecon.environments.environment import VerySimpleEnvironment
from agents.greedy_agent import GreedyAgent
from engines.unreal_cv_wrapper import UnrealCVWrapper
from utils import Timer


def run(environment, agent):
    environment.reset()

    # Setup loop timer
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        timer = Timer()
        total_timer = Timer()
        # action_distribution = agent.next_action()
        # action_index = np.random.choice(environment.num_actions(), p=action_distribution)
        action_index = 0
        rospy.loginfo("Next action: {} [{}]".format(action_index, environment.get_action_name(action_index)))
        observation, reward, terminal, info = environment.perform_action(action_index)
        score = info.get("score")
        print("Received reward: {}, score: {}".format(reward, score))
        print("Performing action took {}".format(timer.restart()))
        # pose = environment.get_pose()
        # reward = environment.get_tentative_reward(pose)
        # print("Tentative reward: {}".format(reward))
        # environment._update_map(pose)

        # import ros_utils
        # max_range = 14.0
        # pose = environment.get_pose()
        # rr = agent._perform_raycast(pose, max_range)
        # print("Performing raycast took {}".format(timer.restart()))
        # # Publish point cloud with uncertain voxels
        # if agent._use_surface_only:
        #     pc_xyz = agent._get_uncertain_surface_point_cloud(rr.point_cloud, agent._environment.get_bounding_box())
        # else:
        #     pc_xyz = agent._get_uncertain_point_cloud(rr.point_cloud, agent._environment.get_bounding_box())
        # print("Getting uncertain point cloud took {}".format(timer.restart()))
        # # # print("pc_xyz.shape: ", pc_xyz.shape)
        # uncertain_pc_msg = ros_utils.point_cloud2_numpy_to_ros(pc_xyz)
        # print("Point cloud conversion to ROS msg took {}".format(timer.restart()))
        # uncertain_pc_msg.header.frame_id = 'map'
        # agent._uncertain_pc_pub.publish(uncertain_pc_msg)
        # print("Publishing took {}".format(timer.restart()))
        # print("Loop took {}".format(total_timer.elapsed_seconds()))

        if terminal:
            print("Episode finished. Score: {}. Resetting environment.".format(info["score"]))
            environment.reset()

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('test_agent', anonymous=False)
    bounding_box = math_utils.BoundingBox(
        [-10, -10, 0],
        [+10, +10, +10]
    )
    engine = UnrealCVWrapper(max_depth_viewing_angle=math_utils.degrees_to_radians(70.))
    # environment = Environment(bounding_box)
    environment = VerySimpleEnvironment(bounding_box, engine=engine, clear_size=6.0, random_reset=False)
    # ignore_unknown_voxels = False
    ignore_unknown_voxels = True
    use_surface_only = True
    agent = GreedyAgent(environment, ignore_unknown_voxels=ignore_unknown_voxels, use_surface_only=use_surface_only)
    run(environment, agent)
