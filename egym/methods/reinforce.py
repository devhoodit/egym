from egym.methods.net import Net

import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box
from typing import Tuple

def _normal_distribution(loc, scale):
    return Normal(loc=loc, scale=torch.sigmoid(scale))

class REINFORCE():
    def __init__(self, observation_space, action_space, gamma=0.99, distribution=None) -> None:
        self.gamma = gamma
        if not self.distribution:
            self.distribution = _normal_distribution
        observation_shape = observation_space.shape
        if type(action_space) is Discrete:
            self.action_space = Discrete # action space is discrete
            action_shape = action_space.n
        elif type(action_space) is Box:
            self.action_space = Box # action space is continuous
            action_shape = 2
        else:
            raise TypeError("unknown type rror")
        
        self.policy_net = Net(observation_shape, action_shape)
        self.prob_history = []
        self.reward_history = []

    def get_history(self):
        return self.reward_history, self.prob_history

    def save_reward_history(self, reward):
        self.reward_history.append(reward)

    def save_prob_history(self, prob):
        self.prob_history.append(prob)
    
    def reset_history(self):
        self.reward_history = []
        self.prob_history = []

    def sample_action(self, state: np.ndarray) -> Tuple[int, float]:
        """sample action from current state

        Args:
            state (np.ndarray): state

        Returns:
            Tuple[int, float]: selected action, prob to select action
        """
        x = torch.from_numpy(state.astype(np.float32))
        output = self.policy_net(x)
        if self.action_space is Discrete:
            prob = F.softmax(output, dim=0)
            prob_distribution = Categorical(prob)
            action = prob_distribution.sample()
            return action.item(), prob[action]
        else:
            prob_distribution = self.distribution(output[0], output[1]) # output[1] can be negative, must pretreatment this
            action = prob_distribution.sample()
            return action, prob_distribution.log_prob(action)
    
    def train_net(self):
        self.policy_net.optim.zero_grad()
        Gt = 0
        for r, prob in zip(self.reward_history[::-1], self.prob_history[::-1]):
            Gt = r + self.gamma * Gt
            loss = -Gt * prob
            loss.backward()
        self.policy_net.optim.step()
        self.reset_history()