from egym.sampling.samplingPolicy import SamplingPolicy
import math

class EgreedyPolicy(SamplingPolicy):    
    def annealing_step(self):
        raise NotImplementedError()
    
    def current(self):
        raise NotImplementedError()
    
class StepEgreedyPolicy(EgreedyPolicy):
    def __init__(self, initial: float, stride: float, minimum: float) -> None:
        self.cur_e = initial
        self.stride = stride
        self.minimum = minimum

    def annealing_step(self):
        self.cur_e -= self.stride

    def current(self):
        return max(self.cur_e, self.minimum)
    
class ExponentialEgreedyPolicy(EgreedyPolicy):
    def __init__(self, initial: float, decay: int, minimum: float):
        self.initial = initial
        self.decay = decay
        self.minimum = minimum
        self.iter_count = 0

    def annealing_step(self):
        self.iter_count += 1
    
    def cureent(self):
        return self.minimum + math.exp(-1. * self.iter_count / self.decay) * (self.initial - self.minimum)