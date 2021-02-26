import gym
import gym_mouse
import time
import agent_assets.agent_models as am
from agent_assets.parallel_tools import ParallelTrainer
from agent_assets import tools
import agent_assets.A_hparameters as hp
import argparse
from tensorflow.profiler.experimental import Profile
from datetime import timedelta

ENVIRONMENT = 'mouseUnity-v0'

env_kwargs = [
    dict(
        ip='localhost',
        port=7777
    ),
    dict(
        ip='localhost',
        port=7778
    ),
    dict(
        ip='localhost',
        port=7779
    ),
    dict(
        ip='localhost',
        port=7780
    ),
    dict(
        ip='localhost',
        port=7781
    ),
    dict(
        ip='localhost',
        port=7782
    ),
    dict(
        ip='localhost',
        port=7783
    ),
    dict(
        ip='localhost',
        port=7784
    )
]
env_names = [ENVIRONMENT]*len(env_kwargs)

hp.CLASSIC = False

model_f = am.unity_conv_vmpo

hp.Actor_activation = 'tanh'

evaluate_f = tools.evaluate_unity

parser = argparse.ArgumentParser()
parser.add_argument('--step', dest='total_steps',default=100000, type=int)
parser.add_argument('-n','--logname', dest='log_name',default=None)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
parser.add_argument('-mf','--mixedfloat', dest='mixed_float', 
                    action='store_true',default=False)
parser.add_argument('-l','--load', dest='load', default=None)
args = parser.parse_args()

total_steps = int(args.total_steps)

hp.Algorithm = 'V-MPO'

hp.Batch_size = 4
hp.Buf.N = 4
hp.k_train_step = 1
hp.Q_discount = 0.9
hp.Target_update = 100

hp.Model_save = 2000
hp.histogram = 1000

hp.lr['common'].halt_steps = 0
hp.lr['common'].start = 1e-4
hp.lr['common'].end = 1e-4
hp.lr['common'].nsteps = 2e4
hp.lr['common'].epsilon = 1e-5
hp.lr['common'].grad_clip = 0.1

hp.lr['encoder'].halt_steps = 0
hp.lr['encoder'].start = 1e-5
hp.lr['encoder'].end = 1e-5
hp.lr['encoder'].nsteps = 1e6
hp.lr['encoder'].epsilon = 1e-5
hp.lr['encoder'].grad_clip = None

hp.lr['forward'] = hp.lr['encoder']
hp.lr['inverse'] = hp.lr['encoder']

hp.VMPO_eps_eta = 1e-1
hp.VMPO_eps_alpha_mu = 1e-2
hp.VMPO_eps_alpha_sig = 1e-5

hp.IQN_ENABLE = False

hp.ICM_ENABLE = False
hp.ICM_intrinsic = 1.0
hp.ICM_loss_forward_weight = 0.2

# For benchmark
st = time.time()

p_trainer = ParallelTrainer(
    model_f=model_f,
    m_dir = args.load,
    log_name=args.log_name,
    mixed_float = args.mixed_float,
    env_names=env_names,
    env_kwargs=env_kwargs,
)

if args.profile:
    p_trainer.train_n_steps(20)
    
    with Profile(f'logs/{args.log_name}'):
        p_trainer.train_n_steps(3)
    
    p_trainer.train_n_steps(total_steps-23, evaluate_f)


else :
    p_trainer.train_n_steps(total_steps)

p_trainer.save_and_evaluate(evaluate_f)
d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')

