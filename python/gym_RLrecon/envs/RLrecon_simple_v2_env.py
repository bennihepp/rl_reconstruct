import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import rospy
from RLrecon import math_utils
from RLrecon.environment import SimpleV1Environment
from RLrecon.engine.dummy_engine import DummyEngine
from RLrecon.engine.unreal_cv_wrapper import UnrealCVWrapper


class RLreconSimpleV1Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RLreconSimpleV1Env, self).__init__()
        rospy.init_node('gym_RLrecon_simple_node', anonymous=False)
        world_bounding_box = math_utils.BoundingBox(
            [-20, -20,   0],
            [+20, +20, +20],
        )
        score_bounding_box = math_utils.BoundingBox(
            [-10, -10, -10],
            [+10, +10, +10]
        )
        engine = DummyEngine()
        self._environment = SimpleV1Environment(
            world_bounding_box, engine=engine, random_reset=True,
            clear_size=6.0, filter_depth_map=False,
            score_bounding_box=score_bounding_box)
        self.action_space = spaces.Discrete(self._environment.num_actions())
        # self.observation_space = spaces.Box(
        #     0.0,
        #     1.0,
        #     shape=self._environment.observation_shape()
        # )
        self._obs_level = 3
        self._obs_size_x = 16
        self._obs_size_y = self._obs_size_x
        self._obs_size_z = self._obs_size_x
        self.observation_space = spaces.Box(
            0.0,
            1.0,
            shape=(self._obs_size_x, self._obs_size_y, self._obs_size_z, 1)
        )

    def _configure(self, client_id, remotes):
        address = '127.0.0.1'
        port = 9900 + client_id
        engine = UnrealCVWrapper(
            address=address,
            port=port,
            max_depth_viewing_angle=math_utils.degrees_to_radians(70.))
        self._environment._engine = engine

    def _get_observation(self):
        level = self._obs_level
        size_x = self._obs_size_x
        size_y = self._obs_size_y
        size_z = self._obs_size_z
        center = self._environment.get_location()
        orientation_rpy = self._environment.get_orientation_rpy()
        # We query a subvolume of the occupancy map so that z-axis is aligned with gravity (roll = pitch = 0)
        # query_orientation_rpy = np.array([0, 0, orientation_rpy[2]])
        query_orientation_rpy = np.array([0, orientation_rpy[1], orientation_rpy[2]])
        # TODO: Should be exposed in environment
        res = self._environment.get_mapper().perform_query_subvolume_rpy(
            center, query_orientation_rpy, level, size_x, size_y, size_z)
        occupancies = np.asarray(res.occupancies, dtype=np.float32)
        occupancies_3d = np.reshape(occupancies, (size_x, size_y, size_z, 1))
        location = self._environment.get_location()
        # orientation_rpy = self._environment.get_orientation_rpy()
        orientation_quat = self._environment.get_orientation_quat()
        return [location, orientation_quat, occupancies_3d]

    def _step(self, action):
        self._previous_state = self._current_state
        new_state, reward, terminal, info = self._environment.perform_action(action, self._current_state)
        self._current_state = new_state
        observation = self._get_observation()
        print("score:", info["score"])
        return observation, reward, terminal, info

    def _reset(self):
        self._environment.reset()
        self._previous_state = self._environment.get_pose()
        self._current_state = self._environment.get_pose()
        observation = self._get_observation()
        return observation

    def _render(self, mode='human', close=False):
        pass
