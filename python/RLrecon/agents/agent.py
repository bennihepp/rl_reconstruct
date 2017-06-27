import numpy as np


class BaseAgent(object):

    def __init__(self, temperature=1.0):
        self._temperature = temperature

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        assert(temperature > 0)
        self._temperature = temperature

    def _values_to_prob_distribution(self, values):
        """Convert a list of values/rewards to a probability distribution with the currently set 'temperature'"""
        exp_values = np.exp(values / self._temperature)
        prob_dist = exp_values / np.sum(exp_values)
        return prob_dist
