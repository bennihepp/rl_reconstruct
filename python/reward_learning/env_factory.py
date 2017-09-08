from __future__ import print_function

import numpy as np
import yaml
import rospy
from RLrecon import math_utils
from pybh.unreal.unreal_cv_wrapper import UnrealCVWrapper
import RLrecon.environments.environment as RLenvironment


def get_environment_class_by_name(environment_name):
    if environment_name == "HorizontalEnvironment":
        environment_class = RLenvironment.HorizontalEnvironment
    elif environment_name == "SimpleV0Environment":
        environment_class = RLenvironment.SimpleV0Environment
    elif environment_name == "SimpleV2Environment":
        environment_class = RLenvironment.SimpleV2Environment
    else:
        raise NotImplementedError("Unknown environment class: {}".format(environment_name))
    return environment_class


def create_environment(environment_class, client_id=0):
    rospy.init_node('env_factory', anonymous=False)
    if environment_class == RLenvironment.HorizontalEnvironment:
        world_bounding_box = math_utils.BoundingBox(
            [-18, -20,   0],
            [+23, +18, +20],
        )
        score_bounding_box = math_utils.BoundingBox(
            [-14, -17.5, -0.5],
            [+19.5, +15, +5]
        )
    elif environment_class == RLenvironment.SimpleV2Environment \
        or environment_class == RLenvironment.SimpleV0Environment:
        world_bounding_box = math_utils.BoundingBox(
            [-20, -20,   0],
            [+20, +20, +20],
        )
        score_bounding_box = math_utils.BoundingBox(
            [-3, -3, -0.5],
            [+3, +3, +5]
        )
    else:
        raise NotImplementedError("Environment of type {} is not supported".format(environment_class))
    # score_bounding_box = math_utils.BoundingBox(
    #     np.array([-np.inf, -np.inf, -np.inf]),
    #     np.array([np.inf, np.inf, np.inf]))
    # score_bounding_box = None
    clear_size = -1.0
    address = '127.0.0.1'
    port = 9900 + client_id
    engine = UnrealCVWrapper(
        address=address,
        port=port,
        image_scale_factor=1.0)
    environment = environment_class(
        world_bounding_box, engine=engine, random_reset=True,
        clear_size=clear_size, filter_depth_map=False,
        score_bounding_box=score_bounding_box)
    return environment


def create_environment_from_yaml(yaml_file, client_id=0):
    if type(yaml_file) == str:
        with open(yaml_file, "r") as fin:
            return create_environment_from_yaml(fin, client_id)
    config = yaml.load(yaml_file)
    rospy.init_node('env_factory', anonymous=False)
    environment_name = config["environment_class"]
    environment_class = get_environment_class_by_name(environment_name)
    assert(len(config["world_bounding_box"]) == 2)
    world_bounding_box = math_utils.BoundingBox(
        config["world_bounding_box"][0],
        config["world_bounding_box"][1])
    score_bounding_box = math_utils.BoundingBox(
        config["score_bounding_box"][0],
        config["score_bounding_box"][1])
    collision_obs_level = config["collision"]["obs_level"]
    collision_obs_sizes = config["collision"]["obs_sizes"]

    clear_size = -1.0
    address = '127.0.0.1'
    port = 9900 + client_id
    engine = UnrealCVWrapper(
        address=address,
        port=port,
        image_scale_factor=1.0)
    environment = environment_class(
        world_bounding_box, engine=engine, random_reset=True,
        clear_size=clear_size, filter_depth_map=False,
        score_bounding_box=score_bounding_box,
        collision_obs_level=collision_obs_level,
        collision_obs_sizes=collision_obs_sizes)

    assert (np.allclose(engine.get_width(), config["camera"]["width"]))
    assert (np.allclose(engine.get_height(), config["camera"]["height"]))
    assert (np.allclose(engine.get_horizontal_field_of_view_degrees(), config["camera"]["fov"]))
    mapper_info = environment.get_mapper().perform_info()
    assert(np.allclose(mapper_info.resolution, config["octomap"]["voxel_size"]))
    assert(np.allclose(mapper_info.max_range, config["octomap"]["max_range"]))
    assert(mapper_info.use_only_surface_voxels_for_score == config["octomap"]["use_only_surface_voxels_for_score"])
    assert(mapper_info.binary_surface_voxels_filename == config["octomap"]["binary_surface_voxels_filename"])

    return environment
