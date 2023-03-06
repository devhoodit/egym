from __future__ import annotations
from egym.sampling.samplingPolicy import SamplingPolicy
from egym.sampling.egreedy import StepEgreedyPolicy
from egym.sampling.greedy import GreedyPolicy
from egym.methods.sarsa import SARSA

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
        self.agent = SARSA(self.observation_space, self.action_space, gamma=self.gamma, sampling_policy=self.sampling_policy)
        reward_history = []

        for n_epi in range(episodes):
            done = False
            s, _ = self.env.reset()
            score = 0.0

            while not done:
                a = self.agent.sample_action(s)
                s_prime, reward, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                transition = (s, a, reward, s_prime)
                self.agent.update_table(transition)
                s = s_prime
                score += reward
            self.sampling_policy.step()
            reward_history.append(score)

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
        state_history = []
        s, _ = self.env.reset()
        score = 0.
        done = False
        state_history.append((s, score))
        while not done:
            a = self.agent.eval_action(s)
            s_prime, reward, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated
            s = s_prime
            score += reward
            state_history.append((s, score))
        return state_history
        


