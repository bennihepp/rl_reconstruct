from __future__ import print_function

import numpy as np
import rospy
from RLrecon import math_utils
from RLrecon.engines.unreal_cv_wrapper import UnrealCVWrapper
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
        image_scale_factor=0.25)
    environment = environment_class(
        world_bounding_box, engine=engine, random_reset=True,
        clear_size=clear_size, filter_depth_map=False,
        score_bounding_box=score_bounding_box)
    return environment
