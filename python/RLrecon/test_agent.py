import numpy as np
import rospy
from tf import transformations
from agents.greedy_agent import GreedyAgent
from environment import Environment, SimpleEnvironment
import math_utils


def run(environment, agent):
    environment.initialize(clear_size=5)
    environment.start()

    # Setup loop timer
    rate = rospy.Rate(0.5)

    while environment.is_running() and not rospy.is_shutdown():
        state = environment.get_state()
        print("Current state: {}".format(state))
        action_distribution = agent.next_action(state)
        action_index = np.random.choice(environment.num_actions(), p=action_distribution)
        # action_index = 0
        rospy.loginfo("Next action: {} [{}]".format(action_index, environment.get_action_name(action_index)))
        new_state, reward = environment.perform_action(action_index)
        print("Received reward: {}".format(reward))
        # reward = environment.get_tentative_reward(state)
        # print("Tentative reward: {}".format(reward))
        # environment._update_map(state)

        import ros_utils
        max_range = 14.0
        rr = agent._perform_raycast(new_state, max_range)
        # Publish point cloud with uncertain voxels
        if agent._use_surface_only:
            pc_xyz = agent._get_uncertain_surface_point_cloud(rr.point_cloud, agent._environment.get_bounding_box())
        else:
            pc_xyz = agent._get_uncertain_point_cloud(rr.point_cloud, agent._environment.get_bounding_box())
        # # print("pc_xyz.shape: ", pc_xyz.shape)
        uncertain_pc_msg = ros_utils.point_cloud2_numpy_to_ros(pc_xyz)
        uncertain_pc_msg.header.frame_id = 'map'
        agent._uncertain_pc_pub.publish(uncertain_pc_msg)

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('test_agent', anonymous=False)
    bounding_box = math_utils.BoundingBox(
        [-10, -10, 0],
        [+10, +10, +10]
    )
    # environment = Environment(bounding_box)
    environment = SimpleEnvironment(bounding_box)
    # ignore_unknown_voxels = False
    ignore_unknown_voxels = True
    use_surface_only = True
    agent = GreedyAgent(environment, ignore_unknown_voxels=ignore_unknown_voxels, use_surface_only=use_surface_only)
    run(environment, agent)
