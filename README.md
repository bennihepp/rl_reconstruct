# LearnToScore

This repo contains code to learn a utility function for Next-Best-View exploration with a depth camera.

The *cpp* folder contains a modified [Octomap](https://github.com/OctoMap)) implementation that allows to compute
a viewpoint score when ground-truth voxels are known and also exposes functions to extract a 3D grid of the occupancy map
on multiple scales relative to a defined camera pose.

A dataset of such 3D grids together with the ground-truth viewpoint score can be used to train a 3D ConvNet
to regress a viewpoint utility score. The code to learn this 3D ConvNet is in *python/reward_learning* and written
with *TensorFlow*. A script to record such datasets and evaluate the utility score regressor is also in this folder.

A useful OpenGL renderer written with [Glumpy](https://github.com/glumpy/glumpy) can be found in *python/renderer*.

The communication between the modified *Octomap* and the TensorFlow code is done with [ROS](http://www.ros.org/).
The images from the OpenGL renderer can be grabbed with [ZeroMQ](http://zeromq.org/).
Images can also be grabbed from Unreal Engine using a modified version of [UnrealCV](https://github.com/unrealcv/unrealcv)
which can be found in the *pybh* submodule.

## Task description and 3D grid extraction
![Task](https://raw.githubusercontent.com/bennihepp/rl_reconstruct/master/images/task_description.png?token=AA957cTV6NVNdQg4lB1pFUgemN905PCBks5aZQcjwA%3D%3D)
![3D grid extraction](https://raw.githubusercontent.com/bennihepp/rl_reconstruct/master/images/occupancy_grid_extraction.png?token=AA957RzmMyFZycueNxZ-p0VXFmCi9xc4ks5aZQcxwA%3D%3D)


## Some results from real a scene
![Example exploration 1](https://raw.githubusercontent.com/bennihepp/rl_reconstruct/master/images/scene_20_video_0_steps_200.png?token=AA957UrKhwSC12wmQh_NMi0HIhJDnoxRks5aZQd_wA%3D%3D)
![Example exploration 2](https://raw.githubusercontent.com/bennihepp/rl_reconstruct/master/images/scene_buildings7_episode_0_steps_200.png?token=AA957e-V8afikIFhTGqdyx6CHSePxJt2ks5aZQd_wA%3D%3D)
