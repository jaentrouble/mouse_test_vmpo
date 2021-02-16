import gym
import gym_mouse
import time
import numpy as np
from Agent import Player
import agent_assets.agent_models as am
from agent_assets import tools
import agent_assets.A_hparameters as hp
from tqdm import tqdm
import argparse
import os
import sys
from tensorflow.profiler.experimental import Profile
from datetime import timedelta

ENVIRONMENT = 'mouseUnity-v0'

env_kwargs = dict(
    ip='localhost',
    port = 7777,
)

model_f = am.unity_conv_ddpg

hp.Actor_activation = 'tanh'

evaluate_f = tools.evaluate_unity

parser = argparse.ArgumentParser()
parser.add_argument('-r','--render', dest='render',action='store_true', default=False)
parser.add_argument('--step', dest='total_steps',default=100000, type=int)
parser.add_argument('-n','--logname', dest='log_name',default=None)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
# parser.add_argument('-lr', dest='lr', default=1e-5, type=float)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-l','--load', dest='load', default=None)
args = parser.parse_args()

vid_type = 'mp4'
total_steps = int(args.total_steps)
my_tqdm = tqdm(total=total_steps, dynamic_ncols=True)


hp.Model_save = 30000
hp.Learn_start = 20000

hp.lr['actor'].halt_steps = 0
hp.lr['actor'].start = 1e-5
hp.lr['actor'].end = 1e-5
hp.lr['actor'].nsteps = 1e6
hp.lr['actor'].epsilon = 1e-3
hp.lr['actor'].grad_clip = None

hp.lr['critic'].halt_steps = 0
hp.lr['critic'].start = 1e-4
hp.lr['critic'].end = 1e-4
hp.lr['critic'].nsteps = 1e6
hp.lr['critic'].epsilon = 1e-3
hp.lr['critic'].grad_clip = None

hp.lr['encoder'].halt_steps = 0
hp.lr['encoder'].start = 1e-5
hp.lr['encoder'].end = 1e-5
hp.lr['encoder'].nsteps = 1e6
hp.lr['encoder'].epsilon = 1e-5
hp.lr['encoder'].grad_clip = None

hp.lr['forward'] = hp.lr['encoder']
hp.lr['inverse'] = hp.lr['encoder']

hp.OUP_stddev_start = 0.2
hp.OUP_stddev_end = 0.05
hp.OUP_stddev_nstep = 2e5
hp.OUP_stddev_nstep = int(hp.OUP_stddev_nstep)
hp.OUP_noise_max = 0.5

hp.IQN_ENABLE = False

hp.ICM_ENABLE = False
hp.ICM_intrinsic = 1.0
hp.ICM_loss_forward_weight = 0.2

hp.Target_update_tau = 1e-2

hp.Buf.N = 1

# For benchmark
st = time.time()

need_to_eval = False

env = gym.make(ENVIRONMENT, **env_kwargs)
bef_o = env.reset()

player = Player(
    observation_space= env.observation_space, 
    action_space= env.action_space, 
    model_f= model_f,
    tqdm= my_tqdm,
    m_dir=args.load,
    log_name= args.log_name,
    mixed_float=args.mixed_float,
)

if args.render :
    env.render()

if args.profile:
    # Warm up
    for step in range(hp.Learn_start+20):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

    with Profile(f'logs/{args.log_name}'):
        for step in range(5):
            action = player.act(bef_o)
            aft_o,r,d,i = env.step(action)
            player.step(bef_o,action,r,d,i)
            if d :
                bef_o = env.reset()
            else:
                bef_o = aft_o
            if args.render :
                env.render()
    remaining_steps = total_steps - hp.Learn_start - 25
    for step in range(remaining_steps):
        if ((hp.Learn_start + 25 + step) % hp.Model_save) == 0 :
            need_to_eval = True
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            if need_to_eval:
                player.save_model()
                score = evaluate_f(player, env, vid_type)
                print('eval_score:{0}'.format(score))
                need_to_eval = False

            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

else :
    for step in range(total_steps):
        if (step>0) and ((step % hp.Model_save) == 0) :
            need_to_eval = True
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            if need_to_eval:
                player.save_model()
                score = evaluate_f(player, env, vid_type)
                print('eval_score:{0}'.format(score))
                need_to_eval = False

            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

player.save_model()
score = evaluate_f(player, env, vid_type)
print('eval_score:{0}'.format(score))
d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')
my_tqdm.close()

