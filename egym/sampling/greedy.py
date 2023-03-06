from egym.sampling.samplingPolicy import SamplingPolicy

class GreedyPolicy(SamplingPolicy):
    def sample_action(self, values):
        raise NotImplementedError("Must implement sample_action method for SamplingPolicy")