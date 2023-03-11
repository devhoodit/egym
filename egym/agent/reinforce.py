from __future__ import annotations
import numpy as np
from gymnasium.core import Env
import torch

from egym.methods.reinforce import REINFORCE

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
        reward_history = []

        for n_epi in range(episode):
            done = False
            s, _ = self.env.reset()
            score = 0.0
            while not done:
                action, prob = self.method.sample_action(s)
                s_prime, reward, terminate, truncated, _ = self.env.step(action)
                done = terminate or truncated
                self.method.save_reward_history(reward)
                self.method.save_prob_history(prob)
                s = s_prime
                score += reward
            reward_history.append(score)
            self.method.train_net()

            if n_epi % self.print_cycle == 0 and not silent:
                print(self.print_format.format(
                    episode=n_epi,
                    average_reward=np.average(reward_history),
                    max_reward=max(reward_history),
                    min_reward=min(reward_history),
                    std_reword=np.std(reward_history)
                ))
                reward_history = []

            if n_epi % self.save_history_cycle == 0:
                pass # implement needed for plot

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