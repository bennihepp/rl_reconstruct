from gym.envs.registration import register

register(
    id='RLrecon-v0',
    entry_point='gym_RLrecon.envs:RLreconEnv',
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)

register(
    id='RLrecon-simple-v0',
    entry_point='gym_RLrecon.envs:RLreconSimpleV0Env',
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)

register(
    id='RLrecon-simple-v1',
    entry_point='gym_RLrecon.envs:RLreconSimpleV1Env',
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)

register(
    id='RLrecon-simple-v2',
    entry_point='gym_RLrecon.envs:RLreconSimpleV2Env',
    tags={'unreal': True},
    timestep_limit=80,
    nondeterministic = True,
    # reward_threshold=1.0,
)

register(
    id='RLrecon-simple-v3',
    entry_point='gym_RLrecon.envs:RLreconSimpleV3Env',
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)

register(
    id='RLrecon-very-simple-v0',
    entry_point='gym_RLrecon.envs:RLreconVerySimpleEnv',
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)

register(
    id='RLrecon-dummy-v0',
    entry_point='gym_RLrecon.envs:RLreconDummyEnv',
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)


from envs import RLreconEnvWrapper
from RLrecon.environments.fixed_environment import FixedEnvironmentV0
from RLrecon.environments.fixed_environment import FixedEnvironmentV1
from RLrecon.environments.fixed_environment import FixedEnvironmentV2
from RLrecon.environments.yaw_only_environment import YawOnlyEnvironmentV1
from RLrecon.environments.yaw_only_environment import YawOnlyEnvironmentV2


register(
    id='RLrecon-fixed-v0',
    entry_point='gym_RLrecon.envs:RLreconEnvWrapper',
    kwargs={"environment_class": FixedEnvironmentV0},
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)


register(
    id='RLrecon-fixed-v1',
    entry_point='gym_RLrecon.envs:RLreconEnvWrapper',
    kwargs={"environment_class": FixedEnvironmentV1},
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)


register(
    id='RLrecon-fixed-v2',
    entry_point='gym_RLrecon.envs:RLreconEnvWrapper',
    kwargs={"environment_class": FixedEnvironmentV2},
    tags={'unreal': True},
    timestep_limit=50,
    nondeterministic = True,
    # reward_threshold=1.0,
)


register(
    id='RLrecon-yaw-only-v1',
    entry_point='gym_RLrecon.envs:RLreconEnvWrapper',
    kwargs={"environment_class": YawOnlyEnvironmentV1},
    tags={'unreal': True},
    timestep_limit=20,
    nondeterministic = True,
    # reward_threshold=1.0,
)


register(
    id='RLrecon-yaw-only-v2',
    entry_point='gym_RLrecon.envs:RLreconEnvWrapper',
    kwargs={"environment_class": YawOnlyEnvironmentV2},
    tags={'unreal': True},
    timestep_limit=20,
    nondeterministic = True,
    # reward_threshold=1.0,
)
