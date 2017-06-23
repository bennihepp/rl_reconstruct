import numpy as np


class Environment(object):

  class State(object):

    def __init__(self, location, orientation_rpy):
      self._location = location
      self._orientation_rpy = orientation_rpy

    def location(self):
      return self._location

    def orientation_rpy(self):
      return self._orientation_rpy

  def __init__(self,
               engine=None,
               mapper=None,
               move_distance=2.0,
               yaw_amount=np.pi / 10.,
               pitch_amount=np.pi / 5.):
    if engine is None:
      from engine.unreal_cv_wrapper import UnrealCVWrapper
      engine = UnrealCVWrapper()
    self._engine = engine
    if mapper is None:
      from mapping.octomap_ext_mapper import OctomapExtMapper
      mapper = OctomapExtMapper()
    self._mapper = mapper
    self._action_map = [
      self.move_left,
      self.move_right,
      self.move_down,
      self.move_up,
      self.move_backward,
      self.move_forward,
      self.yaw_clockwise,
      self.yaw_counter_clockwise,
      # self.pitch_down,
      # self.pitch_up,
    ]
    self._move_distance = move_distance
    self._yaw_amount = yaw_amount
    self._pitch_amount = pitch_amount
    self._current_state = self.get_state()

  def get_state(self):
    location = self._engine.get_location()
    orientation_rpy = self._engine.get_orientation_rpy()
    return self.State(location, orientation_rpy)

  def reset_state(self, state):
    self._update_state(state)

  def _update_state(self, new_state):
    self._current_state = new_state
    self._engine.set_location(self._current_state.location())
    self._engine.set_orientation_rpy(self._current_state.orientation_rpy())

  def _get_depth_point_cloud(self, state):
    # TODO
    #point_cloud = self._engine.get_depth_point_cloud_rpy(state.location(), state.orientation_rpy())
    point_cloud = None
    return point_cloud

  def _update_map(self, state):
    point_cloud = self._get_point_cloud(state)
    self._mapper.update_map_rpy(state.location(), state.orientation_rpy(), point_cloud)

  def _simulate_action(self, action_index):
    new_state = self._action_map[action_index](self._current_state)
    return new_state

  def num_actions(self):
    return len(self._action_map)

  def perform_action(self, action_index):
    new_state = self._action_map[action_index](self._current_state)
    self.update_state(new_state)
    self.update_map(new_state)
    return new_state

  def get_action_name(self, action_index):
    return self._action_map[action_index].__name__

  def _move(self, state, offset):
    new_location = state.location() + offset
    return self.State(new_location, state.orientation_rpy())

  def move_left(self, state):
    return self._move(state, np.array([0, -self._move_distance, 0]))

  def move_right(self, state):
    return self._move(state, np.array([0, +self._move_distance, 0]))

  def move_down(self, state):
    return self._move(state, np.array([0, 0, -self._move_distance]))

  def move_up(self, state):
    return self._move(state, np.array([0, 0, +self._move_distance]))

  def move_backward(self, state):
    return self._move(state, np.array([-self._move_distance, 0, 0]))

  def move_forward(self, state):
    return self._move(state, np.array([+self._move_distance, 0, 0]))

  def yaw_clockwise(self, state):
    [roll, pitch, yaw] = state.orientation_rpy()
    new_yaw = yaw - self._yaw_amount
    return self.State(state.location, np.array([roll, pitch, new_yaw]))

  def yaw_counter_clockwise(self, state):
    [roll, pitch, yaw] = state.orientation_rpy()
    new_yaw = yaw + self._yaw_amount
    return self.State(state.location, np.array([roll, pitch, new_yaw]))

  def pitch_down(self, state):
    [roll, pitch, yaw] = state.orientation_rpy()
    new_pitch = pitch - self._pitch_amount
    return self.State(state.location, np.array([roll, new_pitch, yaw]))

  def pitch_up(self, state):
    [roll, pitch, yaw] = self._engine.get_orientation_rpy()
    new_pitch = pitch + self._pitch_amount
    return self.State(state.location, np.array([roll, new_pitch, yaw]))

  def get_tentative_reward(self, state, action_index):
    tentative_state = self._simulate_action(state, action_index)
    rr = self._mapper.perform_raycast_rpy(
      tentative_state.location(),
      tentative_state.orientation_rpy()
    )
    return rr.expected_reward
