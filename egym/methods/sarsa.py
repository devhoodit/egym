from egym.sampling.samplingPolicy import SamplingPolicy
from egym.sampling.egreedy import StepEgreedyPolicy

from typing import Sequence, Tuple
from gymnasium.spaces.discrete import Discrete
import numpy as np
import random

# for discrete observation space
# Q-learning method

class SARSA():
    def __init__(self, observation_space, action_space: Discrete, gamma=0.1, sampling_policy: SamplingPolicy=StepEgreedyPolicy(1, 0.01, 0.01)) -> None:
        self.sampling_policy = sampling_policy
        if type(observation_space) is Discrete:
            observation_shape = (observation_space.n,)
        elif type(observation_space) is Tuple:
            space = []
            for discrete in observation_space:
                space.append(discrete.n)
            observation_shape = tuple(space)
        else:
            raise TypeError("unknown type")
        if type(action_space) is not Discrete:
            raise TypeError("action space parameter only can get Discrete()")
        self.action_size = action_space.n
        self.qtable: np.ndarray = np.zeros((observation_shape + (self.action_size, )))
        self.trajectory = [] # s, a, r, s_prime => state, action, reward, state_prime (next state)

        self.gamma = gamma
        self.lr = 0.1

    def sample_action(self, state: Sequence):
        rand = random.random()
        if rand < self.sampling_policy.current():
            return random.randint(0, self.action_size-1)
        else:
            actions = self.qtable.__getitem__(state)
            action = np.argmax(actions)
            return action
        
    def update_table(self, transition: Tuple):
        s, a, r, s_prime = transition
        if type(s) is int:
            s = tuple([s])
            s_prime = tuple([s_prime])
        a = tuple([a])
        a_prime = self.sample_action(s_prime)
        a_prime = tuple([a_prime])
        self.qtable.__setitem__(s+a, self.qtable.__getitem__(s+a) + self.lr * (r + self.gamma * self.qtable.__getitem__(s_prime+a_prime) - self.qtable.__getitem__(s+a)))
    
    def show_table(self):
        print(self.qtable)

