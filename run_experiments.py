import argparse
import datetime
import multiprocessing
import subprocess

MJCTimeStepLookUp = {
    "HalfCheetah-v2": 3e6,
    "Hopper-v2": 1e6,
    "Walker2d-v2": 3e6,
    "Humanoid-v2": 10e6,
    "Ant-v2": 3e6,
    "InvertedPendulum-v2": 1e6
}

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='HalfCheetah-v2')
parser.add_argument('--alg', type=str, default='td3', choices={'td3', 'sac'})
parser.add_argument('--seeds5to9', dest='seeds5to9', action='store_true')
parser.add_argument('--total_timesteps', type=int, default=None)
parser.set_defaults(seeds5to9=False)

args = parser.parse_args()
params = vars(args)

if params['total_timesteps']:
    total_timesteps = params['total_timesteps']
elif params['env'] in MJCTimeStepLookUp:
    total_timesteps = int(MJCTimeStepLookUp[params['env']])
else:
    total_timesteps = int(1e6)

date_time_string = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')
experiment_id = 'runs/' + params['env'] + '/' + params['alg'] + '/' + date_time_string + '/seed'
num_experiments = 5
seeds_5to9 = params['seeds5to9']
lower = 0
upper = num_experiments

main_experiment = ["python", "train_agent.py", "--env", params['env'], "--alg", params['alg'], "--n_evals", str(10), "--total_timesteps", str(total_timesteps), "--seed"]

if seeds_5to9:
    lower += 5
    upper += 5

all_experiments = [main_experiment + [str(i)] + ["--experiment_name"] + [experiment_id + str(i)] for i in range(lower, upper)]

def run_experiment(spec):
    subprocess.run(spec, check=True)

def run_all_experiments(specs):
    pool = multiprocessing.Pool()
    pool.map(run_experiment, specs)

run_all_experiments(all_experiments)
