from egym.methods.reinforce import REINFORCE

from gymnasium.core import Env

class REINFORCEAgent():
    def __init__(self, env: Env) -> None:
        self.env = env
        
        # user set values
        self.gamma = 0.99
        
        # print values
        self.print_cycle = 100
        self.print_format = "Episode: {episode} Reward - Avg: {average_reward} \tMax: {max_reward} \tMin: {min_reward} \tStd: {std_reword}"

        #plot values
        self.save_history_cycle = 1