import gym
import turtlebot_gym

env = gym.make('drl-nav-v0')
obs = env.reset()
print("Initial observation:", obs)

action = env.action_space.sample()  # Take a random action
obs, reward, done, info = env.step(action)
print("Observation after action:", obs)