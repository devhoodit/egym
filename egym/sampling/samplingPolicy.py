from abc import ABCMeta, abstractmethod

class SamplingPolicy(metaclass=ABCMeta):
    @abstractmethod
    def annealing_step(self):
        pass

    @abstractmethod
    def current(self):
        pass
