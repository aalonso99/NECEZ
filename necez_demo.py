import torch
import os
import yaml

from models import MuZeroNECCartNet
from memory import GameRecord, Memory
from envs import cartpole_env

from mcts import search, MinMax
from utils import convert_to_int

# Constants
CONFIG_PATH = os.path.join("configs", "config-cartpole-nec.yaml")
GAME_TYPE = "cartpole"
DEBUG = True


# Load environment and env parameters
config = yaml.safe_load(open(CONFIG_PATH, "r"))
config["debug"] = DEBUG
env = cartpole_env.make_env(config, render_mode="human")

config["full_image_size"] = env.full_image_size
config["action_size"] = env.action_space.n
obs_size = config["obs_size"]
obs_size = obs_size[0]

# Set torch parameters
device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

# Load MuZero model
nec_muzero_class = MuZeroNECCartNet

config["action_dim"] = 1
nec_mu_net = nec_muzero_class(config["action_size"], obs_size, config, weights_path="demo_model/demo_weights.pt")
nec_mu_net.load_dnd("demo_model/demo_dnd.pickle")

# print(len(nec_mu_net.pred_net.dnd.memory_table))

minmax = MinMax() # Used to normalize values in tree search

# Start environment
obs = env.reset()
done = False

# Main loop
while not done:

    obs = convert_to_int(obs[0], GAME_TYPE)

    tree = search(
        config, nec_mu_net, obs, minmax, device=device
    )

    action = tree.pick_game_action(temperature=0.0)
    print(action)

    obs, _, done, _ = env.step(action)

    env.render()


env.close()