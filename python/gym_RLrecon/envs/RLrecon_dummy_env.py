import gym
import numpy as np
import rospy
from RLrecon import math_utils
from RLrecon.engines.dummy_engine import DummyEngine
from RLrecon.engines.unreal_cv_wrapper import UnrealCVWrapper
from RLrecon.environments.environment import SimpleEnvironment
from gym import spaces


class RLreconDummyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RLreconDummyEnv, self).__init__()
        rospy.init_node('gym_RLrecon_dummy_node', anonymous=False)
        bounding_box = math_utils.BoundingBox(
            [-10, -10, 0],
            [+10, +10, +10]
        )
        engine = DummyEngine()
        self._environment = SimpleEnvironment(bounding_box, engine=engine, random_reset=True, clear_size=6.0)
        self.action_space = spaces.Discrete(self._environment.num_actions())
        self.observation_space = spaces.Box(
            -1000,
            +1000,
            shape=(2,)
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
        obs = np.array([
            self._environment._get_orbit_angle(self._previous_state),
            self._environment._get_orbit_angle(self._current_state),
        ], dtype=np.float32)
        return obs

    def _step(self, action):
        self._previous_state = self._current_state
        new_state, reward, score = self._environment.perform_action(action, self._current_state)
        self._current_state = new_state
        observation = self._get_observation()
        terminal = score >= 4000
        info = {
            "score": score,
        }
        return observation, reward, terminal, info

    def _reset(self):
        self._environment.reset()
        self._previous_state = self._environment.get_pose()
        self._current_state = self._environment.get_pose()
        observation = self._get_observation()
        # observation = self._current_state.location()
        return observation

    def _render(self, mode='human', close=False):
        pass
