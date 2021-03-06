import gym
import gym_mouse
import time
from tqdm import trange
import numpy as np
import cv2
import argparse
from agent_assets.Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools
import agent_assets.A_hparameters as hp

parser = argparse.ArgumentParser()
parser.add_argument('-l','--load', dest='load', required=True,)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)

args = parser.parse_args()


model_f = am.unity_conv_vmpo

evaluate_f = tools.evaluate_unity

env_kwargs = dict(
    ip='localhost',
    port = 7777,
)

hp.IQN_ENABLE = False
hp.ICM_ENABLE = False


st = time.time()
env = gym.make('mouseUnity-v0', **env_kwargs)
env.reset()
player = Player(
    observation_space=env.observation_space,
    action_space= env.action_space, 
    model_f= model_f,
    mixed_float=args.mixed_float,
    m_dir=args.load,
)

score = evaluate_f(player, env, 'mp4')
