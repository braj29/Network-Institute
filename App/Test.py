
import gym
import griddly
from griddly import gd
import time

env = gym.make('GDY-Labyrinth-v0', player_observer_type=gd.ObserverType.VECTOR,
               global_observer_type=gd.ObserverType.SPRITE_2D)

env.reset()
# env = gym.make('CarRacing-v0')
#from pettingzoo.mpe import simple_v2
#from multiagent import policy
#env = simple_v2.env()
#print(f"Action Space : {env.action_spaces['agent_0']}")
print(f"Action Space : {env.action_space.n}")
print(f"Action Space Sample : {env.action_space.sample()}")
#print(f"Action Space Meanings: {env.unwrapped.get_action_meanings()}")
print(f"Action Space Keys : {env.unwrapped.get_keys_to_action()}")
#print(f"Observation Space : {env.observation_space}")
print(f"Reward Range : {env.reward_range}")
print(env.game.build_valid_action_trees())
#help(env.action_space)

#help(env.unwrapped)
#help(env.action_spaces['agent_0'])