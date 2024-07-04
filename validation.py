import torch
import os
import yaml
import pickle
import itertools

from models import MuZeroNECCartNet, MuZeroCartNet
from envs import cartpole_env

from mcts import search, MinMax

# algorithms = ['EZ']
algorithms = ['NECEZ']
consistencies = ['consistency_05', 'consistency_1', 'consistency_2', 'consistency_5']
NUM_RUNS = 10

# Load environment and env parameters
# CONFIG_PATH = os.path.join("configs", "config-cartpole.yaml")
CONFIG_PATH = os.path.join("configs", "config-cartpole-nec.yaml")
config = yaml.safe_load(open(CONFIG_PATH, "r"))
config["debug"] = False
env = cartpole_env.make_env(config)

config["full_image_size"] = env.full_image_size
config["action_size"] = env.action_space.n
obs_size = config["obs_size"]
obs_size = obs_size[0]

run_log_dirs = { f"{algo}_{consistency}":
                    [ f"./runs/{algo}/{consistency}/{run_log_dir}"
                        for run_log_dir in os.listdir(f"./runs/{algo}/{consistency}") ] 
                    for algo, consistency in itertools.product(algorithms, consistencies)
                    if os.path.exists(f"./runs/{algo}") and os.path.exists(f"./runs/{algo}/{consistency}")
                }

def run_game(env, config, muzero_network, max_iters=200, device=torch.device("cpu")):
    minmax = MinMax()
    obs = env.reset(return_render=False)
    done = False
    score = 0
    i = 0
    while not done:
        tree = search(
            config, muzero_network, obs, minmax, device=device
        )
        action = tree.pick_game_action(temperature=0.0)
        obs, reward, terminated, truncated, info = env.step(action, return_render=False)
        i += 1
        done = terminated or truncated or i >= max_iters
        score += reward
    return score

def load_model(algo, run_log_dir, config):
    weights_path = os.path.join(run_log_dir, "latest_model_dict.pt")
    print(f"Loading weights from {weights_path}")

    if algo == "EZ":
        muzero_network = MuZeroCartNet(config["action_size"], obs_size, config)
    elif algo == "NECEZ":
        muzero_network = MuZeroNECCartNet(config["action_size"], obs_size, config)
        muzero_network.load_dnd(os.path.join(run_log_dir, "latest_dnd.pickle"))

    muzero_network.load_state_dict(torch.load(weights_path))
    muzero_network.eval()

    return muzero_network


scores = { f"{algo}_{consistency}": { run_log_dir : [] 
                                      for run_log_dir 
                                      in run_log_dirs[f"{algo}_{consistency}"] }
           for algo, consistency 
           in itertools.product(algorithms, consistencies) 
           if os.path.exists(f"./runs/{algo}") and os.path.exists(f"./runs/{algo}/{consistency}")
        }

for algo_consistency in run_log_dirs.keys():
    print(f"Running {algo_consistency}")

    algo = algo_consistency.split("_")[0]
    consistency = '_'.join(algo_consistency.split("_")[1:])

    for run_log_dir in run_log_dirs[algo_consistency]:
        muzero_network = load_model(algo, run_log_dir, config)

        for i in range(NUM_RUNS):
            print(f"Running game {i}")
            score = run_game(env, config, muzero_network)
            scores[algo_consistency][run_log_dir].append(score)
            print(f"Score: {score}")

        print(f"Average score: {sum(scores[algo_consistency][run_log_dir]) / NUM_RUNS}")
            
# pickle.dump(scores, open("validation_scores_EZ.pkl", "wb"))
pickle.dump(scores, open("validation_scores_NECEZ.pkl", "wb"))