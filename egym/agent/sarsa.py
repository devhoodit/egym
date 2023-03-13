from __future__ import annotations
from egym.sampling.samplingPolicy import SamplingPolicy
from egym.sampling.egreedy import StepEgreedyPolicy
from egym.sampling.greedy import GreedyPolicy
from egym.methods.sarsa import SARSA
from egym.formatting.formatting import wrap_print_cycle

from gymnasium.core import Env
import numpy as np

class SARSAAgent:
    def __init__(self, env: Env) -> None:
        # must set in initial values
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # user set values
        self.gamma = 0.1
        self.sampling_policy = StepEgreedyPolicy(1, 0.0001, 0.1)

        # print values
        self.print_cycle = 100
        self.print_format = "Episode: {episode} Reward - Avg: {average_reward} \tMax: {max_reward} \tMin: {min_reward} \tStd: {std_reword}"

        #plot values
        self.save_history_cycle = 1 # save all episode history
        

    def set_gamma(self, value: float) -> SARSAAgent:
        self.gamma = value
        return self
    
    def set_sampling_policy(self, sampling_policy: SamplingPolicy) -> SARSAAgent:
        self.sampling_policy = sampling_policy
        return self
    
    def set_episode_save_cycle(self, value: int) -> SARSAAgent:
        self.save_history_cycle = value
        return self
    
    def is_valid(self):
        return True
    
    def set_print_format(self, format: str) -> SARSAAgent:
        self.print_format = format
        return
    
    def train(self, episodes: int, silent=False):
        if not self.is_valid():
            raise NotImplementedError("set all variables")
        self.method = SARSA(self.observation_space, self.action_space, gamma=self.gamma, sampling_policy=self.sampling_policy)
        
        batch_reward_history = []
        batch_iter_length_history = []
        reward_history = []
        iter_length_history = []

        for n_epi in range(episodes):
            done = False
            s, _ = self.env.reset()
            score = 0.0
            iter_length = 0
            while not done:
                iter_length += 1
                a = self.method.sample_action(s)
                s_prime, reward, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                transition = (s, a, reward, s_prime)
                self.method.update_table(transition)
                s = s_prime
                score += reward
            self.sampling_policy.step()
            
            batch_reward_history.append(score)
            batch_iter_length_history.append(score)

            if n_epi % self.print_cycle == 0 and not silent and n_epi != 0:
                print(wrap_print_cycle(self.print_format, n_epi, batch_reward_history, batch_iter_length_history))
                batch_reward_history = []
                batch_iter_length_history = []

            if n_epi % self.save_history_cycle == 0:
                reward_history.append(score)
                iter_length_history.append(iter_length)

    def eval(self):
        state_history = []
        s, _ = self.env.reset()
        score = 0.
        done = False
        state_history.append((s, score))
        while not done:
            a = self.method.eval_action(s)
            s_prime, reward, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated
            s = s_prime
            score += reward
            state_history.append((s, score))
        return state_history
        


