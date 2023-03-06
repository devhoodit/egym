from abc import ABCMeta, abstractmethod

class SamplingPolicy(metaclass=ABCMeta):
    @abstractmethod
    def sample_action(self, values):
        pass
