from egym.sampling.egreedy import EgreedyPolicy, StepEgreedyPolicy

class RandomPolicy(EgreedyPolicy):
    def __init__(self) -> None:
        self.policy = StepEgreedyPolicy(1, 0, 1)
    
    def annealing_step(self) -> None:
        pass

    def current(self) -> float:
        return self.policy.current()