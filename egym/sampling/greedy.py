from egym.sampling.samplingPolicy import SamplingPolicy

import numpy as np

class GreedyPolicy(SamplingPolicy):
    def sample_action(self, values):
        return np.argmax(values)