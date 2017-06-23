import numpy as np


class GreedyAgent(object):

  def __init__(self, environment):
    self._environment = environment

  def next_action(self, state):
    best_reward = -np.inf
    best_action_index = -1
    for action_index in xrange(self._environment.num_actions()):
      print("Testing action {}".format(action_index))
      reward = self._environment.get_tentative_reward(state, action_index)
      if reward > best_reward:
        best_reward = reward
        best_action_index = action_index
    if best_action_index < 0:
      print("WARNING: Could not determine a useful action. Choosing random action.")
      best_action_index = np.random.randint(0, self._environment.num_actions())
    return best_action_index
