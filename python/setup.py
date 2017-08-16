#!/usr/bin/env python

from distutils.core import setup

gym_available = False
try:
    import gym
    gym_available = True
except:
    pass

packages = ['RLrecon', 'reward_learning']
additional_kwargs = {}

if gym_available:
    packages.append('gym_RLrecon')
    additional_kwargs['install_requires'] = 'gym'

setup(name='RLrecon',
      version='0.1',
      description='Reinforcement learning for 3D reconstruction',
      author='Benjamin Hepp',
      author_email='benjamin.hepp@posteo.de',
      license='BSD 3 License',
      packages=packages,
      **additional_kwargs
)
