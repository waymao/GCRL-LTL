# %%
import argparse
import random
import time

import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.traj_buffer import TrajectoryBuffer
from rl.callbacks import CollectTrajectoryCallback
from envs import ZonesEnv, ZoneRandomGoalTrajEnv
from envs.utils import get_zone_vector

timeout = 100

# %%
device = torch.device('cuda')

# %%
env = ZoneRandomGoalTrajEnv(
    env=gym.make('Zones-8-v0'), 
    primitives_path='models/primitives', 
    zones_representation=get_zone_vector(),
    # use_primitves=True,
    rewards=[0, 1],
    use_primitves=True,
    device=device,
)


policy = PPO('MultiInputPolicy', env, verbose=1, device=device)
policy.learn(total_timesteps=1000000)

# %%
obs = env.reset()

# %%
for i in range(1000):
    done = False
    obs, info = env.reset()
    print('obs', obs, 'obs_space', env.observation_space)
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        # env.render()
        # events
        # print(env.get_events())

        # 
        time.sleep(0.01)

# %%



