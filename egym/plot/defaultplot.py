import matplotlib.pyplot as plt
from typing import List

class DefaultPlot():
    def __init__(self) -> None:
        pass
    
    def __overplot(self, data):
        pass
    
    def cumulative_reward(self, data: List):
        plt.figure(figsize=(10,8))
        plt.ylabel('Cumulative Reward')
        plt.xlabel('Step')
        data = data.sort()
        for category in data:
            self.__overplot(category)
            