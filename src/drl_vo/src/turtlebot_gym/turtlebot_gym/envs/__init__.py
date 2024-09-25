from gym.envs.registration import register
from .drl_nav_env import DRLNavEnv

# Register drl_nav env 
register(
  id='drl-nav-v0',
  entry_point='turtlebot_gym.envs:DRLNavEnv'
  )


