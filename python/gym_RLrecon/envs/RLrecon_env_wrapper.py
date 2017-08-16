from __future__ import print_function
import gym
import numpy as np
import rospy
from RLrecon import math_utils
from RLrecon.engines.dummy_engine import DummyEngine
from RLrecon.engines.unreal_cv_wrapper import UnrealCVWrapper
from gym import spaces


class RLreconEnvWrapper(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, environment_class):
        super(RLreconEnvWrapper, self).__init__()
        rospy.init_node('gym_RLrecon_env_wrapper', anonymous=False)
        world_bounding_box = math_utils.BoundingBox(
            [-20, -20,   0],
            [+20, +20, +20],
        )
        score_bounding_box = math_utils.BoundingBox(
            [-3, -3, -0.5],
            [+3, +3, +5]
        )
        engine = DummyEngine()
        self._environment = environment_class(
            world_bounding_box, engine=engine, random_reset=True,
            clear_size=6.0, filter_depth_map=False,
            score_bounding_box=score_bounding_box)
        self.action_space = spaces.Discrete(self._environment.get_num_of_actions())
        # self.observation_space = spaces.Box(
        #     0.0,
        #     1.0,
        #     shape=self._environment.observation_shape()
        # )
        # self._obs_level = 3
        # self._obs_size_x = 16
        # self._obs_level = 4
        # self._obs_size_x = 8
        # self._obs_size_y = self._obs_size_x
        # self._obs_size_z = self._obs_size_x
        # self.observation_space = spaces.Box(
        #     0.0,
        #     1.0,
        #     shape=(self._obs_size_x, self._obs_size_y, self._obs_size_z, 2)
        # )
        # self.observation_space = spaces.Box(
        #     -1000,
        #     +1000,
        #     shape=(14,)
        # )
        spaces_tuple = []
        for obs_shape in self._environment.get_observation_shapes():
            spaces_tuple.append(spaces.Box(
                -np.finfo(np.float32).max,
                +np.finfo(np.float32).max,
                shape=obs_shape
            ))
        self.observation_space = spaces.Tuple(spaces_tuple)

    def _configure(self, client_id, remotes):
        address = '127.0.0.1'
        port = 9900 + client_id
        engine = UnrealCVWrapper(
            address=address,
            port=port,
            image_scale_factor=0.25,
            max_depth_viewing_angle=math_utils.degrees_to_radians(70.))
        self._environment._engine = engine

    def _step(self, action):
        self._previous_pose = self._current_pose
        observation, reward, terminal, info = self._environment.perform_action(action, self._current_pose)
        self._current_pose = self._environment.get_pose()
        print("Reward:", reward, "Score:", info["score"])
        return observation, reward, terminal, info

    def _reset(self):
        observation = self._environment.reset()
        self._previous_pose = self._environment.get_pose()
        self._current_pose = self._environment.get_pose()
        return observation

    def _render(self, mode='human', close=False):
        pass
