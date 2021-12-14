# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:25:41 2020

@author: Hang Yu
"""


import DQN
import gym, random, pickle, os.path, math, glob, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

from atari_wrappers import make_atari, wrap_deepmind,LazyFrames
from IPython.display import clear_output
from tensorboardX import SummaryWriter

env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, scale = False, frame_stack=True)

episodes = 1

action_space = env.action_space
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
state_channel = env.observation_space.shape[2]
sum_rewards = []
agent=DQN.DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = True, lr = 0.001)
agent.DQN.load_state_dict(torch.load('PongGame.pth'))
episode_rewards = []
frame = env.reset()

for i in range(episodes):

    
    while(True):
        env.render()
        time.sleep(0.02)
        state_tensor = agent.observe(frame)
        action = agent.act(state_tensor, 0, False).argmax()

        next_frame, reward, done, _ = env.step(action)
        frame = next_frame
        episode_rewards.append( reward)
        if done:
            print(" reward: %5f, episode: %4d" % (np.sum(episode_rewards), i))
            sum_rewards.append(np.sum(episode_rewards))
            frame = env.reset()
            break
env.close()
plt.plot([i for i in range(episodes)],sum_rewards)
plt.savefig("Training Curve.jpg")