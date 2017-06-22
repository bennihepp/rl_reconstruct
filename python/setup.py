#!/usr/bin/env python

from distutils.core import setup

setup(name='RLrecon',
	version='0.1',
	description='Reinforcement learning for 3D reconstruction',
	author='Benjamin Hepp',
	author_email='benjamin.hepp@posteo.de',
	license='BSD 3 License',
	packages=['RLrecon', 'RLrecon.engine'],
)
