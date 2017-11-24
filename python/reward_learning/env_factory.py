from __future__ import print_function
import six
import numpy as np
import yaml
from multiprocessing.pool import ThreadPool
import rospy
import gym
import gym.spaces
from pybh import math_utils, utils
import RLrecon.environments.environment as RLenvironment
from RLrecon.mapping.octomap_ext_mapper import OctomapExtMapper


class MultiEnvironmentWrapper(object):

    def __init__(self, environments, auto_reset=True):
        self._environments = environments
        self._auto_reset = auto_reset
        self._num_resets = [0] * len(self._environments)
        self._episode_lengths = [0] * len(self._environments)
        self._prev_episode_lengths = [0] * len(self._environments)
        self._total_steps = [0] * len(self._environments)

    @property
    def num_resets(self):
        return self._num_resets

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def prev_episode_lengths(self):
        return self._prev_episode_lengths

    @property
    def episode_lengths(self):
        return self._episode_lengths

    @property
    def environments(self):
        return self._environments

    @property
    def action_space(self):
        return self._environments[0].action_space

    @property
    def observation_space(self):
        return self._environments[0].observation_space

    @property
    def reset_interval(self):
        return self._environments[0].reset_interval

    @reset_interval.setter
    def reset_interval(self, reset_interval):
        for env in self._environments:
            env.reset_interval = reset_interval

    @property
    def reset_score_threshold(self):
        return self._environments[0].reset_score_threshold

    @reset_score_threshold.setter
    def reset_score_threshold(self, reset_score_threshold):
        for env in self._environments:
            env.reset_score_threshold = reset_score_threshold

    def _convert_list_to_batch_observation(self, observation_list):
        if isinstance(observation_list[0], dict):
            observation_batch = {}
            for key in observation_list[0]:
                if isinstance(observation_list[0][key], np.ndarray):
                    observation_batch[key] = np.empty((len(observation_list),) + observation_list[0][key].shape)
                    for i in range(len(observation_list)):
                        obs = observation_list[i][key]
                        observation_batch[key][i, ...] = obs
                else:
                    observation_batch[key] = []
                    for obs in observation_list:
                        observation_batch[key].append(obs)
        elif isinstance(observation_list[0], np.ndarray):
            observation_batch = np.empty((len(observation_list),) + observation_list[0].shape)
            for i in range(len(observation_list)):
                observation_batch[i, ...] = observation_list[i]
        else:
            raise NotImplementedError("Only dict or list observations are supported")
        return observation_batch

    def _single_reset(self, env_index):
        env = self._environments[env_index]
        observation = env.reset()
        self._num_resets[env_index] += 1
        if self._episode_lengths[env_index] >= 190:
            print("Episode ran for {} steps".format(self._episode_lengths[env_index]))
            assert False
        self._prev_episode_lengths[env_index] = self._episode_lengths[env_index]
        self._episode_lengths[env_index] = 0
        return observation

    def reset(self):
        observation_list = [None] * len(self._environments)
        for i in range(len(self._environments)):
            observation = self._single_reset(i)
            observation_list[i] = observation
        observation_batch = self._convert_list_to_batch_observation(observation_list)
        return observation_batch

    def _single_step(self, env_index, action_index):
        env = self._environments[env_index]
        observation, reward, done, info = env.step(action_index)
        self._episode_lengths[env_index] += 1
        self._total_steps[env_index] += 1
        if done and self._auto_reset:
            observation = self._single_reset(env_index)
        return observation, reward, done, info

    def step(self, action_index_batch):
        observation_list = [None] * len(self._environments)
        reward_batch = np.empty((len(self._environments),), dtype=np.float32)
        done_batch = np.empty((len(self._environments),), dtype=np.int32)
        info_batch = [None] * len(self._environments)
        for i in range(len(self._environments)):
            observation, reward, done, info = self._single_step(i, action_index_batch[i])
            observation_list[i] = observation
            reward_batch[i] = reward
            done_batch[i] = done
            info_batch[i] = info
        with utils.get_time_meter("env_factory").measure("convert_observation_to_batch"):
            observation_batch = self._convert_list_to_batch_observation(observation_list)
        return observation_batch, reward_batch, done_batch, info_batch


class PooledMultiEnvironmentWrapper(MultiEnvironmentWrapper):

    def __init__(self, environments, auto_reset=True):
        super(PooledMultiEnvironmentWrapper, self).__init__(environments, auto_reset)
        self._pool = ThreadPool(len(self._environments))

    # def _single_reset(self, env_index):
    #     env = self._environments[env_index]
    #     observation = env.reset()
    #     return observation

    def reset(self):
        with utils.get_time_meter("pooled_multi_env").measure("parallel_reset"):
            observation_list = self._pool.map(lambda i: self._single_reset(i), range(len(self._environments)))
        observation_batch = self._convert_list_to_batch_observation(observation_list)
        return observation_batch

    # def _single_step(self, env_index, action_index):
    #     env = self._environments[env_index]
    #     observation, reward, done, info = env.step(action_index)
    #     if done and self._auto_reset:
    #         env.reset()
    #     return observation, reward, done, info

    def step(self, action_index_batch):
        with utils.get_time_meter("pooled_multi_env").measure("parallel_step"):
            results = self._pool.map(lambda x: self._single_step(x[0], x[1]), enumerate(action_index_batch))
        observation_list = []
        reward_batch = np.empty((len(self._environments),), dtype=np.float32)
        done_batch = np.empty((len(self._environments),), dtype=np.int32)
        info_list = []
        for i, result in enumerate(results):
            observation, reward, done, info = result
            observation_list.append(observation)
            reward_batch[i] = reward
            done_batch[i] = done
            info_list.append(info)
        with utils.get_time_meter("pooled_multi_env").measure("convert_observation_to_batch"):
            observation_batch = self._convert_list_to_batch_observation(observation_list)
        return observation_batch, reward_batch, done_batch, info_list


class OpenAIEnvironmentWrapper(object):

    def __init__(self, environment, reset_score_threshold, reset_interval,
                 obs_levels, obs_sizes, axis_mode=0, forward_factor=3 / 8.,
                 downsample_to_grid=False):
        self._environment = environment
        self._intrinsics = self._environment.get_engine().get_intrinsics()
        self._reset_score_threshold = reset_score_threshold
        self._reset_interval = reset_interval
        self._num_steps = 0
        self._obs_levels = obs_levels
        self._obs_sizes = obs_sizes
        self._axis_mode = axis_mode
        self._forward_factor = forward_factor
        self._downsample_to_grid = downsample_to_grid
        mapper_info = self._environment.get_mapper().perform_info()
        self._map_resolution = mapper_info.resolution
        self._action_space = gym.spaces.Discrete(self._environment.get_num_of_actions())
        self._observation_space = gym.spaces.Dict({
            "in_grid_3d": gym.spaces.Box(low=0.0, high=1.0, shape=[len(obs_levels)] + list(obs_sizes)),
            "allowed_action:": gym.spaces.Box(low=0, high=1, shape=(self._environment.get_num_of_actions(),))
        })
        self._scores = []

    @property
    def base(self):
        return self._environment

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reset_interval(self):
        return self._reset_interval

    @reset_interval.setter
    def reset_interval(self, reset_interval):
        self._reset_interval = reset_interval

    @property
    def reset_score_threshold(self):
        return self._reset_score_threshold

    @reset_score_threshold.setter
    def reset_score_threshold(self, reset_score_threshold):
        self._reset_score_threshold = reset_score_threshold

    def _query_octomap(self, pose):
        grid_3ds = None
        for k in range(len(self._obs_levels)):
            obs_level = self._obs_levels[k]
            obs_size_x = self._obs_sizes[0]
            obs_size_y = self._obs_sizes[1]
            obs_size_z = self._obs_sizes[2]

            obs_resolution = self._map_resolution * (2 ** obs_level)
            offset_x = obs_resolution * obs_size_x * self._forward_factor
            offset_vec = math_utils.rotate_vector_with_rpy(pose.orientation_rpy(), [offset_x, 0, 0])
            query_location = pose.location() + offset_vec
            query_pose = self._environment.Pose(query_location, pose.orientation_rpy())

            res = self._environment.get_mapper().perform_query_subvolume_rpy(
                query_pose.location(), query_pose.orientation_rpy(),
                obs_level, obs_size_x, obs_size_y, obs_size_z, self._axis_mode)
            occupancies = np.asarray(res.occupancies, dtype=np.float32)
            occupancies_3d = np.reshape(occupancies, (obs_size_x, obs_size_y, obs_size_z))
            observation_certainties = np.asarray(res.observation_certainties, dtype=np.float32)
            observation_certainties_3d = np.reshape(observation_certainties, (obs_size_x, obs_size_y, obs_size_z))

            grid_3d = np.stack([occupancies_3d, observation_certainties_3d], axis=-1)
            if grid_3ds is None:
                grid_3ds = grid_3d
            else:
                grid_3ds = np.concatenate([grid_3ds, grid_3d], axis=-1)

        return grid_3ds

    def _perform_action(self, action_index):
        current_pose = self._environment.get_pose()
        assert not self._environment.is_action_colliding(current_pose, action_index, verbose=True)
        new_pose = self._environment.simulate_action_on_pose(current_pose, action_index)
        self._environment.set_pose(new_pose)
        # TODO: Should use Environment.perform_action()
        # engine_pose = (new_pose.location(), new_pose.orientation_rpy())
        # self._environment.get_engine().set_pose_rpy(engine_pose)
        return new_pose

    def _update_state(self, pose):
        depth_image = self._environment.get_engine().get_depth_image()
        result = self._environment.get_mapper().perform_insert_depth_map_rpy(
            pose.location(), pose.orientation_rpy(),
            depth_image, self._intrinsics, downsample_to_grid=self._downsample_to_grid, simulate=False)
        return result

    def get_observation(self, pose):
        in_grid_3ds = self._query_octomap(pose)
        allowed_actions = np.ones((self._environment.get_num_of_actions(),), dtype=np.int)
        for action_index in range(self._environment.get_num_of_actions()):
            if self._environment.is_action_colliding(pose, action_index):
                allowed_actions[action_index] = 0
        observation = {"in_grid_3d": in_grid_3ds,
                       "allowed_action": allowed_actions}
        return observation

    def _compute_score_auc(self, scores):
        score_auc = np.sum(scores)
        remaining_timesteps = self._reset_interval - len(scores)
        score_auc += remaining_timesteps * scores[-1]
        score_auc /= self._reset_interval
        score_auc /= self._reset_score_threshold
        return score_auc

    def reset(self):
        # with utils.get_time_meter("openai_wrapper").measure("reset"):
        self._environment.reset()
        self._num_steps = 0
        self._scores = []
        # with utils.get_time_meter("openai_wrapper").measure("get_observation"):
        observation = self.get_observation(self._environment.get_pose())
        return observation

    def step(self, action_index):
        # with utils.get_time_meter("openai_wrapper").measure("perform_action"):
        new_pose = self._perform_action(action_index)
        # with utils.get_time_meter("openai_wrapper").measure("update_state"):
        result = self._update_state(new_pose)
        # TODO: Which reward to take?
        reward = result.probabilistic_reward
        reward = result.normalized_probabilistic_score
        score = result.normalized_probabilistic_score
        self._scores.append(score)
        score_auc = self._compute_score_auc(self._scores)
        # with utils.get_time_meter("openai_wrapper").measure("get_observation"):
        observation = self.get_observation(new_pose)
        info = {"score": score, "score_auc": score_auc}
        self._num_steps += 1
        done = score >= self._reset_score_threshold or self._num_steps >= self._reset_interval
        return observation, reward, done, info


def get_environment_class_by_name(environment_name):
    if environment_name == "HorizontalEnvironment":
        environment_class = RLenvironment.HorizontalEnvironment
    elif environment_name == "EnvironmentNoPitch":
        environment_class = RLenvironment.EnvironmentNoPitch
    elif environment_name == "SimpleV0Environment":
        environment_class = RLenvironment.SimpleV0Environment
    elif environment_name == "SimpleV2Environment":
        environment_class = RLenvironment.SimpleV2Environment
    else:
        raise NotImplementedError("Unknown environment class: {}".format(environment_name))
    return environment_class


# TODO: Deprecated. Should be removed.
# def create_environment(environment_class, client_id=0):
#     rospy.init_node('env_factory', anonymous=False)
#     if environment_class == RLenvironment.HorizontalEnvironment:
#         world_bounding_box = math_utils.BoundingBox(
#             [-18, -20,   0],
#             [+23, +18, +20],
#         )
#         score_bounding_box = math_utils.BoundingBox(
#             [-14, -17.5, -0.5],
#             [+19.5, +15, +5]
#         )
#     elif environment_class == RLenvironment.SimpleV2Environment \
#         or environment_class == RLenvironment.SimpleV0Environment:
#         world_bounding_box = math_utils.BoundingBox(
#             [-20, -20,   0],
#             [+20, +20, +20],
#         )
#         score_bounding_box = math_utils.BoundingBox(
#             [-3, -3, -0.5],
#             [+3, +3, +5]
#         )
#     else:
#         raise NotImplementedError("Environment of type {} is not supported".format(environment_class))
#     # score_bounding_box = math_utils.BoundingBox(
#     #     np.array([-np.inf, -np.inf, -np.inf]),
#     #     np.array([np.inf, np.inf, np.inf]))
#     # score_bounding_box = None
#     clear_size = -1.0
#     address = '127.0.0.1'
#     port = 9900 + client_id
#     engine = UnrealCVWrapper(
#         address=address,
#         port=port,
#         image_scale_factor=1.0)
#     environment = environment_class(
#         world_bounding_box, engine=engine, random_reset=True,
#         clear_size=clear_size, filter_depth_map=False,
#         score_bounding_box=score_bounding_box)
#     return environment


def create_environment_from_yaml_file(yaml_file, client_id=0):
    if type(yaml_file) == str:
        # Use yaml_file as filename
        with open(yaml_file, "r") as fin:
            return create_environment_from_yaml_file(fin, client_id)
    if not type(yaml_file) == dict:
        # Try to use yaml_file as file object
        config = yaml.load(yaml_file)
        if "environment" in config:
            config = config["environment"]
        return create_environment_from_config(config, client_id)


def create_environment_from_config(config, client_id=0, use_openai_wrapper=False,
                                   use_ros=True, prng_seed=None, **env_kwargs):
    environments = create_environments_from_config(config, client_ids=[client_id],
                                                   use_openai_wrapper=use_openai_wrapper,
                                                   use_ros=use_ros,
                                                   prng_seed=prng_seed,
                                                   **env_kwargs)
    return environments[0]


def create_environments_from_config(configs, client_ids, use_openai_wrapper=False,
                                    use_ros=True, prng_seed=None, **env_kwargs):
    if isinstance(client_ids, six.integer_types):
        client_ids = range(client_ids)
    if not isinstance(configs, list):
        configs = [configs] * len(client_ids)
    assert len(configs) == len(client_ids)

    rospy.init_node('RLrecon', anonymous=False)

    environments = []
    for config, client_id in zip(configs, client_ids):
        environment_name = config["environment"]["class"]
        environment_class = get_environment_class_by_name(environment_name)
        assert(len(config["environment"]["world_bounding_box"]) == 2)
        world_bounding_box = math_utils.BoundingBox(
            config["environment"]["world_bounding_box"][0],
            config["environment"]["world_bounding_box"][1])
        start_bounding_box = config["environment"].get("start_bounding_box")
        if start_bounding_box is None:
            start_bounding_box = world_bounding_box
        else:
            start_bounding_box = math_utils.BoundingBox(start_bounding_box[0], start_bounding_box[1])
        # score_bounding_box = math_utils.BoundingBox(
        #     config["environment"]["score_bounding_box"][0],
        #     config["environment"]["score_bounding_box"][1])
        score_bounding_box = math_utils.BoundingBox(
            np.array([-np.inf, -np.inf, -np.inf]),
            np.array([np.inf, np.inf, np.inf]))
        # collision_obs_level = config["environment"]["collision"]["obs_level"]
        # collision_obs_sizes = config["environment"]["collision"]["obs_sizes"]
        collision_bbox_extent = np.asarray(config["environment"]["collision"]["bbox_extent"])
        collision_bbox = math_utils.BoundingBox(-0.5 * collision_bbox_extent, +0.5 * collision_bbox_extent)
        collision_bbox_obs_level = config["environment"]["collision"]["bbox_obs_level"]
        clear_extent = config["environment"]["clear_extent"]
        max_range = config["octomap"]["max_range"]
        if "extra_args" in config["environment"]:
            extra_env_args = config["environment"]["extra_args"]
        else:
            extra_env_args = {}
        extra_env_args.update(env_kwargs)

        engine_name = config.get("engine", "bh_renderer_zmq")

        if engine_name == "bh_renderer_zmq":
            from RLrecon.engines.mesh_renderer_zmq_client import MeshRendererZMQClient
            port = 22222 + client_id
            address = "tcp://localhost:{:d}".format(port)
            print("Creating MeshRendererZMQClient for address {}".format(address))
            engine = MeshRendererZMQClient(address, keep_pose_state=True, max_depth_distance=2 * max_range)
            engine.set_window_active(False)
        elif engine_name == "unreal":
            from pybh.unreal.unreal_cv_wrapper import UnrealCVWrapper
            address = '127.0.0.1'
            port = 9900 + client_id
            engine = UnrealCVWrapper(
                address=address,
                port=port,
                image_scale_factor=1.0)

        if config["camera"].get("stereo", False):
            width = config["camera"]["width"]
            height = config["camera"]["height"]
            stereo_method = config["camera"]["stereo_method"]
            stereo_baseline = config["camera"]["stereo_baseline"]
            min_depth = config["camera"]["min_depth"]
            num_disparities = config["camera"]["num_disparities"]
            block_size = config["camera"]["block_size"]
            from RLrecon.engines.stereo_wrapper import StereoWrapper
            base_engine = engine
            engine = StereoWrapper(base_engine, stereo_method, stereo_baseline, width, height,
                                   min_depth, num_disparities, block_size)

        octomap_server_ext_namespace = "/octomap_server_ext_{:d}/".format(client_id)
        mapper = OctomapExtMapper(octomap_server_ext_namespace=octomap_server_ext_namespace)

        if prng_seed is not None:
            if isinstance(prng_seed, six.integer_types):
                extra_env_args["prng_or_seed"] = prng_seed
            elif isinstance(prng_seed, np.random.RandomState):
                extra_env_args["prng_or_seed"] = prng_seed.randint(np.iinfo(np.int32).max)
            else:
                raise ValueError("Argument prng_seed must be integer or numpy.random.RandomState object")
        environment = environment_class(
            world_bounding_box, engine=engine, mapper=mapper, random_reset=True,
            clear_extent=clear_extent, filter_depth_map=False,
            start_bounding_box=start_bounding_box,
            score_bounding_box=score_bounding_box,
            # collision_obs_level=collision_obs_level,
            # collision_obs_sizes=collision_obs_sizes,
            collision_bbox=collision_bbox,
            collision_bbox_obs_level=collision_bbox_obs_level,
            **extra_env_args)

        assert np.allclose(engine.get_width(), config["camera"]["width"])
        assert np.allclose(engine.get_height(), config["camera"]["height"])
        assert np.allclose(engine.get_horizontal_field_of_view_degrees(), config["camera"]["fov"])
        mapper_info = environment.get_mapper().perform_info()
        assert np.allclose(mapper_info.resolution, config["octomap"]["voxel_size"])
        assert np.allclose(mapper_info.max_range, config["octomap"]["max_range"])
        assert mapper_info.use_only_surface_voxels_for_score == config["octomap"]["use_only_surface_voxels_for_score"]
        import os
        home_path = os.getenv("HOME")
        assert mapper_info.binary_surface_voxels_filename == os.path.join(home_path, config["octomap"]["binary_surface_voxels_filename"])

        if use_ros:
            from RLrecon.environments import ros_environment
            ros_environment.RosEnvironmentHooks(environment, node_namespace="/environment_{:d}/".format(client_id))

        if use_openai_wrapper:
            obs_levels = config["collect_data"]["obs_levels"]
            obs_sizes = config["collect_data"]["obs_sizes"]
            axis_mode = config["collect_data"]["axis_mode"]
            forward_factor = float(config["collect_data"]["forward_factor"])
            downsample_to_grid = config["collect_data"]["downsample_to_grid"]
            reset_interval = config["collect_data"]["reset_interval"]
            reset_score_threshold = float(config["collect_data"]["reset_score_threshold"])
            environment = OpenAIEnvironmentWrapper(environment, reset_score_threshold, reset_interval,
                                                   obs_levels, obs_sizes, axis_mode, forward_factor, downsample_to_grid)

        environments.append(environment)

    return environments
