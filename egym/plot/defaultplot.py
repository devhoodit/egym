import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class DefaultPlot():
    def __init__(self) -> None:
        pass
    
    def __padding_right(self, data):
        max_size = np.max([len(index) for index in data])
        padding_sizes = [max_size - len(index) for index in data]
        for d, pl in zip(data, padding_sizes):
            padding = np.zeros((pl,))
            d[:pl] = padding
        return data
            
    def __overplot(self, data, label: str):
        data = self.__padding_right(data)
        x = np.arange(len(data))
        plt.plot(x, data, label=label)
        plt.fill_between(x, )
    
    def cumulative_reward(self, results: Dict[str, List]):
        plt.figure(figsize=(10,8))
        plt.ylabel('Cumulative Reward')
        plt.xlabel('Step')
        for title, data in results:
            self.__overplot(data, title)
            