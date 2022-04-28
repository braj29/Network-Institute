
# #env = gym.make("CarRacing-v0")
# #play(gym.make("Pong-v4"))

# #play.play(env)
# # def callback(obs_t, obs_tp1, action, rew, done, info):
# #             return [rew,]
# # plotter = play.PlayPlot(callback, 30 * 10, ["reward"])
# # env = gym.make("CarRacing-v0")

# # play.play(env)

# import gym

# from matplotlib import pyplot as plt
# from gym import wrappers
# #import highway_env

# # env = gym.make("parking-v0")
# # env.reset()
# # for _ in range(100):
# #     action = env.action_space.sample()
# #     obs, reward, done, info = env.step(action)
# #     env.render()

# # plt.imshow(env.render(mode="rgb_array"))
# # plt.show()

# # from pettingzoo.mpe import simple_v2
# # #from multiagent import policy
# # env = simple_v2.env(max_cycles = 1000)
# # env.reset()

# # print(env.action_spaces["agent_0"])
# # #a = env.action_spaces["agent_0"].sample()
# # #print(a)
# # #print(help(a))
# # print(env.observation_spaces)
# # for _ in range(1000000):
# #  observation, reward, done, info = env.last()
# #  #action = env.action_space.sample(agent)
# #  print(done)
# #  action = env.action_spaces["agent_0"].sample()
# #  env.step(action)
# #  env.render()


import gym
import griddly
from griddly import gd
import time
import pandas as pd
import numpy as np        
#
# if __name__ == '__main__':
#
#     env = gym.make('GDY-Labyrinth-v0', player_observer_type = gd.ObserverType.VECTOR, global_observer_type = gd.ObserverType.SPRITE_2D, level = 3)
#     #print(dir(gym.make('GDY-Drunk-Dwarf-v0')))
#     state = env.reset()
#     #print(f"Observation Space : {env.observation_space.sample()}")
#
#     # # Replace with your own control algorithm!
#     for _ in range(100000):
#
#         #Qagent = pd.read_pickle(r'q_learning_oracle_3.pkl')
#         #prob = Qagent.action_prob(state)
#         #print(object)
#         #action = np.random.choice([i for i in range(env.action_space.n)],p = prob/sum(prob))
#         action = env.action_space.sample()
#         #print(action)
#         action_space = env.action_space.n
#         #print(action_space)
#         observation_space = env.observation_space.shape
#         #print(observation_space)
#         _, w, h = observation_space
#         print(w, h)
#         #print(env.action_space.sample())
#         obs, reward, done, info = env.step(action)
#         #print(obs)
#         env.render(observer = 'global') # Renders the environment from the perspective of a single player
#         #time.sleep(1)
#         #env.render(observer='global') # Renders the entire environment
#         state = obs
#         if done:
#             env.reset()
#     # print(f"Action Space : {env.action_space.n}")
#     # print(env.observation_space.sample())
#     # print(f"Action Space Sample : {env.action_space.sample()}")
#     # #print(f"Action Space Meanings: {env.unwrapped.get_action_meanings()}")
#     # print(f"Action Space Keys : {env.unwrapped.get_keys_to_action()}")
#     # print(f"Observation Space : {env.observation_space}")
#     # print(env.game.get_object_names())
#     # print(f"Reward Range : {env.reward_range}")
# #     # print(env.action_space)


# #     # # # Replace with your own control algorithm!
# #     # for s in range(1000):
# #     #     obs, reward, done, info = env.step(env.action_space.sample())
# #     #     #print(obs[3].shape)
# #     #     print(obs)
# #     #     w, h = obs[3].shape
# #     #     total_states = w * h
# #     #     print(total_states)
# #     #     #import numpy as np
# #     #     #r = np.where(obs[1]==1)
# #     #     #print(r)
# #     #     #print(type(obs))
# #     #     #env.render() # Renders the environment from the perspective of a single player
# #     #     #print(env.game.get_object_names())
# #     #     env.render(observer='global') # Renders the entire environment

# #     #     if done:
# #     #         env.reset()

import gym
from gym.utils.play import play
import griddly
from griddly import GymWrapperFactory

# This is what to use if you want to use OpenAI gym environments
# wrapper = GymWrapperFactory()
#
# rapper.build_gym_from_yaml('SokobanTutorial', 'sokoban.yaml', level=0)

#Create the Environment
env = gym.make(f'GDY-Labyrinthv1-v0', level = 2)

#Play the game
play(env, fps=10, zoom=1)