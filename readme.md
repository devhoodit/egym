# Egym
Easy to test, use reinforcement learning method for gym environments

# Support Methods
- SARSA (for discrete observation, discrete action)

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