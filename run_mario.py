#Marioを実行するプログラム
#マリオ関連のimport
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

#プロット関連のimport
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import pickle

#vcopt関連のimport
import numpy as np
import numpy.random as nr
from vcopt import vcopt

#エージェント関係のimport 
import random

import cv2
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from util import NoopResetEnv, MaxAndSkipEnv, WarpFrame, ClipRewardEnv, TorchFrame
from util import LazyFrames, FrameStack, PrioritizedReplayBuffer, CNNQNetwork, make_env

import imageio
import gym

#env = make_env()
#ENV_NAME = 'SuperMarioBros-1-1-v0'
#env = gym_super_mario_bros.make(ENV_NAME)
#env = JoypadSpace(env, SIMPLE_MOVEMENT)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('./net.pickle', 'rb') as f:
    net = pickle.load(f)

#print(vars(net))

env = make_env()


#動画の準備
fig = plt.figure()
ims = []

#繰り返し操作してimsに追加 実行確認用
obs = env.reset()

for i in range(500):
    action = nr.randint(1, 7)
    #action = net.act(obs.float().to(device), epsilon=0.0)
    state, reward, done, info = env.step(action) #<-ここに推論が行われた結果としてアクションが入る
    
    #imsに追加
    im = plt.imshow(env.render(mode='rgb_array'))
    ims.append([im])
    
    if done == True:
        break
ani = animation.ArtistAnimation(fig, ims, interval=15, blit=True)
#保存する
ani.save('sample.gif', writer='imagemagick')
print('gif_animation_saved')



#imageio.mimsave('./Mario.gif', ani, 'GIF', **{'duration': 1.0/30.0})
#

'''#動作テスト
#ゲーム環境のリセット
env.reset()


#動画の準備
fig = plt.figure()
ims = []

#繰り返し操作してimsに追加 実行確認用
for i in range(100):
    action = nr.randint(1, 7)
    state, reward, done, info = env.step(action) #<-ここに推論が行われた結果としてアクションが入る
    
    #imsに追加
    im = plt.imshow(env.render(mode='rgb_array'))
    ims.append([im])
    
    if done == True:
        break'''