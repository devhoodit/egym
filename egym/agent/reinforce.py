from __future__ import annotations
import numpy as np
from gymnasium.core import Env
import torch

from egym.methods.reinforce import REINFORCE
from egym.formatting.formatting import wrap_print_cycle
from egym.plot.defaultplot import DefaultPlot

class REINFORCEAgent():
    def __init__(self, env: Env) -> None:
        self.env = env
        
        # user set values
        self.gamma = 0.99
        self.distribution = None
        
        # print values
        self.print_cycle = 100
        self.print_format = "Episode: {episode} Reward - Avg: {average_reward} \tMax: {max_reward} \tMin: {min_reward} \tStd: {std_reword}"

        #plot values
        self.save_history_cycle = 1


    def set_gamma(self, value: float) -> REINFORCEAgent:
        self.gamma = value
        return self
    
    def set_episode_save_cycle(self, value: int) -> REINFORCEAgent:
        self.save_history_cycle = value
        return self
    
    def is_valid(self):
        return True
    
    def set_print_format(self, format: str) -> REINFORCEAgent:
        self.print_format = format
        return self
    
    def set_distribution(self, distribution) -> REINFORCEAgent:
        self.distribution = distribution
        return self
    
    def train(self, episode: int, silent=False):
        if not self.is_valid():
            raise NotImplementedError("set all variables")
        self.method = REINFORCE(self.env.observation_space, self.env.action_space, gamma=self.gamma, distribution=self.distribution)
        self.method.policy_net.train()
        batch_reward_history = []
        batch_iter_length_history = []
        reward_history = []
        iter_length_history = []

        for n_epi in range(episode):
            done = False
            s, _ = self.env.reset()
            score = 0.0
            iter_length = 0
            while not done:
                iter_length += 1
                action, prob = self.method.sample_action(s)
                s_prime, reward, terminate, truncated, _ = self.env.step(action)
                done = terminate or truncated
                self.method.save_reward_history(reward)
                self.method.save_prob_history(prob)
                s = s_prime
                score += reward
            batch_reward_history.append(score)
            batch_iter_length_history.append(iter_length)
            self.method.train_net()

            if n_epi % self.print_cycle == 0 and not silent:
                print(wrap_print_cycle(self.print_format, n_epi, batch_reward_history, batch_iter_length_history))
                batch_reward_history = []
                batch_iter_length_history = []

            if n_epi % self.save_history_cycle == 0:
                reward_history.append(score)
                iter_length_history.append(iter_length)

    def eval(self):
        self.method.policy_net.eval()
        state_history = []
        s, _ = self.env.reset()
        score = 0.0
        done = False
        while not done:
            with torch.no_grad():
                a, _ = self.method.sample_action(s)
            s_prime, reward, terminate, truncated, _ = self.env.step(a)
            done = terminate or truncated
            s = s_prime
            score += reward
            state_history.append((s, score))
        return state_history