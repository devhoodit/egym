from egym.sampling.samplingPolicy import SamplingPolicy

import numpy as np
import random
import math

class EgreedyPolicy(SamplingPolicy):    
    def step(self):
        raise NotImplementedError()
    
    def sample_action(self, values):
        prob = random.random()
        if prob < self.current():
            return random.randint(0, len(values)-1)
        else:
            return np.argmax(values)
    
    
class StepEgreedyPolicy(EgreedyPolicy):
    def __init__(self, initial: float, stride: float, minimum: float) -> None:
        self.cur_e = initial
        self.stride = stride
        self.minimum = minimum

    def step(self):
        self.cur_e -= self.stride

    def current(self):
        return max(self.cur_e, self.minimum)
    
    def sample_action(self, values):
        return super().sample_action(values)
    
class ExponentialEgreedyPolicy(EgreedyPolicy):
    def __init__(self, initial: float, decay: int, minimum: float):
        self.initial = initial
        self.decay = decay
        self.minimum = minimum
        self.iter_count = 0

    def step(self):
        self.iter_count += 1
    
    def cureent(self):
        return self.minimum + math.exp(-1. * self.iter_count / self.decay) * (self.initial - self.minimum)
    
    def sample_action(self, values):
        return super().sample_action(values)