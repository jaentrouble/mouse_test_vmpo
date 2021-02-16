import gym
import gym_mouse
import time
import numpy as np
from Agent import Player
import agent_assets.A_hparameters as hp
import agent_assets.agent_models as am
from agent_assets import tools
from tqdm import tqdm
import argparse
import os
import sys
from tensorflow.profiler.experimental import Profile

ENVIRONMENT = 'mouseCl-v2'

env_kwargs = dict(
    apple_num=10,
    eat_apple = 1.0,
    hit_wall = 0,
)

model_f = am.mouse_eye_brain_model

evaluate_f = tools.evaluate_mouse

parser = argparse.ArgumentParser()
parser.add_argument('-r','--render', dest='render',action='store_true', default=False)
parser.add_argument('-l', dest='load',default=False)
parser.add_argument('--step', dest='loop_steps',default=100000)
parser.add_argument('--loop', dest='total_loops',default=20)
parser.add_argument('--curloop', dest='cur_loop',default=0)
parser.add_argument('-n','--logname', dest='log_name',default=False)
parser.add_argument('--curround', dest='cur_r',default=0)
parser.add_argument('-lb', dest='load_buffer',action='store_true',default=False)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
args = parser.parse_args()

vid_type = 'mp4'
loop_steps = int(args.loop_steps)
total_loops = int(args.total_loops)
cur_loop = int(args.cur_loop)
cur_r = int(args.cur_r)
load_buffer = args.load_buffer

# cur_loop starts from 0
loops_left = total_loops - cur_loop -1

my_tqdm = tqdm(total=loop_steps, dynamic_ncols=True)

hp.epsilon = 1
hp.epsilon_min = 0.1
hp.epsilon_nstep = (total_loops * loop_steps)//2

hp.lr_start = 1e-5
hp.lr_end = 1e-8
hp.lr_nsteps = 1000000

print(f'starting {cur_loop+1}/{total_loops} loop')
if args.render :
    from gym.envs.classic_control.rendering import SimpleImageViewer
    eye_viewer = SimpleImageViewer(maxwidth=1500)
    bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
# For benchmark
st = time.time()

env = gym.make(ENVIRONMENT, **env_kwargs)
bef_o = env.reset()

if args.load :
    player = Player(
        observation_space= env.observation_space, 
        action_space= env.action_space, 
        model_f= model_f,
        tqdm= my_tqdm,
        m_dir= args.load, 
        log_name= args.log_name, 
        start_step= cur_loop*loop_steps, 
        start_round= cur_r, 
        load_buffer= load_buffer,
    )
elif args.log_name:
    # If log directory is explicitely selected
    player = Player(
        observation_space= env.observation_space, 
        action_space= env.action_space, 
        model_f= model_f,
        tqdm= my_tqdm,
        log_name= args.log_name
    )
else :
    player = Player(
        observation_space= env.observation_space,
        action_space= env.action_space, 
        model_f= model_f,
        tqdm= my_tqdm,
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

    with Profile(f'log/{args.log_name}'):
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
    remaining_steps = loop_steps - hp.Learn_start - 25
    for step in range(remaining_steps):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

else :
    for step in range(loop_steps):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

my_tqdm.close()

next_save = player.save_model()
if not args.load:
    save_dir = player.save_dir
else:
    save_dir, _ = os.path.split(args.load)
next_dir = os.path.join(save_dir,str(next_save))
score = evaluate_f(player, gym.make(ENVIRONMENT, **env_kwargs), vid_type)
print('eval_score:{0}'.format(score))
print('{0}steps took {1} sec'.format(loop_steps,time.time()-st))

if loops_left <= 0 :
    sys.exit()
else :
    next_args = []
    next_args.append('python')
    next_args.append(__file__)
    next_args.append('-l')
    next_args.append(next_dir)
    next_args.append('--step')
    next_args.append(str(loop_steps))
    next_args.append('--loop')
    next_args.append(str(total_loops))
    next_args.append('--curloop')
    next_args.append(str(cur_loop+1))
    next_args.append('--logname')
    next_args.append(player.log_name)
    next_args.append('--curround')
    next_args.append(str(player.rounds))
    next_args.append('-lb')
    
    os.execv(sys.executable, next_args)