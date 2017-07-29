from __future__ import print_function

import numpy as np
import rospy
from RLrecon import math_utils
from RLrecon.engines.unreal_cv_wrapper import UnrealCVWrapper
from RLrecon.environments.environment import Environment


def create_environment(environment_class, client_id=0):
    ros_master_uri = "http://localhost:{:d}".format(11911 + client_id)
    rospy.init_node('env_factory', anonymous=False)
    world_bounding_box = math_utils.BoundingBox(
        [-10, -10,   0],
        [+10, +10, +10],
    )
    score_bounding_box = math_utils.BoundingBox(
        [-3, -3, -0.5],
        [+3, +3, +5]
    )
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
