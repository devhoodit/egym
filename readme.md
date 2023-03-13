# Egym
Easy to test, use reinforcement learning method for gym environments

# Support Methods
- SARSA (for discrete observation, discrete action)
- REINFORCE (for continuous observation, discrete or continuous action)

# SARSA (discrete observation, discrete action)
```python
import gymnasium as gym
import egym
env = gym.make("Taxi-v3") # or CliffWalking-v0, FrozenLake-v1

agent = egym.agent.SARSAAgent(env)
agent.train(10000, silent=False) # for print reward infos
agent.agent.show_table()
agent.eval()
```

# REINFORCE (continuous observation, discrete or continuous action)
```python
import gymnasium as gym
import egym
env = gym.make("CartPole-v1") # or Acrobot-v1, MountainCarContinuous-v0, MountainCar-v0, Pendulum-v1, LunarLander-v2

agent = egym.agent.REINFORCEAgent(env)\
    .set_gamma(0.98)
agent.train(5000, silent=False)
agent.eval()
```