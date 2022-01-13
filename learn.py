# coding: UTF-8
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



env = make_env()

#パラメータ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#ハイパーパラメータ

"""
    リプレイバッファの宣言
"""
buffer_size = 100000  #　リプレイバッファに入る経験の最大数
initial_buffer_size = 10000  # 学習を開始する最低限の経験の数
replay_buffer = PrioritizedReplayBuffer(buffer_size)


"""
    ネットワークの宣言
"""
net = CNNQNetwork(env.observation_space.shape, n_action=env.action_space.n).to(device)
target_net = CNNQNetwork(env.observation_space.shape, n_action=env.action_space.n).to(device)
target_update_interval = 2000  # 学習安定化のために用いるターゲットネットワークの同期間隔


"""
    オプティマイザとロス関数の宣言
"""
optimizer = optim.Adam(net.parameters(), lr=1e-4)  # オプティマイザはAdam
loss_func = nn.SmoothL1Loss(reduction='none')  # ロスはSmoothL1loss（別名Huber loss）


"""
    Prioritized Experience Replayのためのパラメータβ
"""
beta_begin = 0.4
beta_end = 1.0
beta_decay = 500000
# beta_beginから始めてbeta_endまでbeta_decayかけて線形に増やす
beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))


"""
    探索のためのパラメータε
"""
epsilon_begin = 1.0
epsilon_end = 0.01
epsilon_decay = 50000
# epsilon_beginから始めてepsilon_endまでepsilon_decayかけて線形に減らす
epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))


"""
    その他のハイパーパラメータ
"""
gamma = 0.9  #　割引率
batch_size = 32
n_episodes = 10  # 学習を行うエピソード数

writer = SummaryWriter('./logs')
step = 0

for episode in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedyで行動を選択
        action = net.act(obs.float().to(device), epsilon_func(step))

        # 環境中で実際に行動
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward

        # リプレイバッファに経験を蓄積
        replay_buffer.push([obs, action, reward, next_obs, done])
        obs = next_obs

        # ネットワークを更新            
        if len(replay_buffer) > initial_buffer_size:
            update(batch_size, beta_func(step))

        # ターゲットネットワークを定期的に同期させる
        if (step + 1) % target_update_interval == 0:
            target_net.load_state_dict(net.state_dict())
        
        step += 1

    print('Episode: {},  Step: {},  Reward: {}'.format(episode + 1, step + 1, total_reward))
    writer.add_scalar('Reward', total_reward, episode)

writer.close()

with open('./net.pickle', 'wb') as f:
    pickle.dump(net, f)