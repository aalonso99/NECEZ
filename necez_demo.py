import torch
import os
import yaml
import pickle

from models import MuZeroNECCartNet
from envs import cartpole_env

from mcts import search, MinMax

import matplotlib.pyplot as plt 
import numpy as np  
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore
import math

def show_images(images, ncols=3):
    n = len(images)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(math.ceil(n/ncols), ncols, i + 1)
        plt.imshow(images[i])

    plt.show(block=False)

    return f

def plot_embedding_space(dnd, latent, knn_indices=None, title=None):
    # Get representation of the latent space from the DND
    X_h = np.asarray( [ r for r,v in dnd.memory_table.values() ] )

    # Plot the latent space
    f = plt.figure()
    plt.scatter(X_h[:, 0], X_h[:, 1], c='blue', s=1)
    plt.scatter(latent[0], latent[1], c='red', s=10)
    if title is not None:
        plt.title(title)
    if knn_indices is not None:
        for i in knn_indices:
            neighbor_latent = dnd.memory_table[i][0]
            plt.scatter(neighbor_latent[0], neighbor_latent[1], c='green', s=5)
    plt.show(block=False)

    return f

def plot_representations(representations, obs_representation, knn_indices):
    # Get representation of the latent space
    X_h = np.asarray(list(representations.values()))

    # Plot the representation space
    f = plt.figure()
    plt.scatter(X_h[:, 0], X_h[:, 1], c='blue', s=1)
    plt.scatter(obs_representation[0], obs_representation[1], c='red', s=10)
    if knn_indices is not None:
        for i in knn_indices:
            neighbor_representation = representations[i]
            plt.scatter(neighbor_representation[0], neighbor_representation[1], c='green', s=5)
    plt.show(block=False)

    return f

# Constants
CONFIG_PATH = os.path.join("configs", "config-cartpole-nec.yaml")
MODEL_DIR = "demo_model_cl05"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "demo_model_dict.pt")
DND_PATH = os.path.join(MODEL_DIR, "demo_dnd.pickle")
DND_ELEMENTS_REPR_PATH = os.path.join(MODEL_DIR, "demo_dnd_elements_representions.pickle")
RAW_OBSERVATIONS_PATH = os.path.join(MODEL_DIR, "raw_observations")
GAME_TYPE = "cartpole"
DEBUG = True
NEIGHBORS_TO_DISPLAY = 1
# RENDER_MODE = "rgb_array"
RENDER_MODE = "human"
ALWAYS_MOVE_LEFT = True

# Load environment and env parameters
config = yaml.safe_load(open(CONFIG_PATH, "r"))
config["debug"] = DEBUG
env = cartpole_env.make_env(config, render_mode=RENDER_MODE)

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
nec_mu_net = nec_muzero_class(config["action_size"], obs_size, config, weights_path=WEIGHTS_PATH)
nec_mu_net.load_dnd(DND_PATH)

def dnd_embedding(representation):
    latent = nec_mu_net.pred_net.fc1(representation)
    latent = torch.relu(latent)
    latent = nec_mu_net.pred_net.fc_value_embedding(latent)
    return latent

minmax = MinMax() # Used to normalize values in tree search

# Fit the outlier detection model on the DND
dnd = nec_mu_net.pred_net.dnd
dnd_points = np.asarray( [ r for r,v in dnd.memory_table.values() ] )
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(dnd_points)
dnd_points_outlier_score = -lof.decision_function(dnd_points)

lof_scaler = StandardScaler()
lof_scaler.fit(dnd_points_outlier_score.reshape(-1,1))

# Read dnd elements representation from directory
dnd_elements_representations = {}
for i in dnd.memory_table:
    obs_pickle_path = os.path.join(RAW_OBSERVATIONS_PATH, str(i))
    with open(obs_pickle_path,'rb') as obs_pickle:  
        # The first element of the pickle is the representation
        dnd_elements_representations[i] = pickle.load(obs_pickle)[0]

with open(DND_ELEMENTS_REPR_PATH, 'wb') as f:
    pickle.dump(dnd_elements_representations, f)

dnd_elements_representations = pickle.load( open(DND_ELEMENTS_REPR_PATH, 'rb') )

# Start environment
obs = env.reset(return_render=True)
done = False
total_reward = 0

step_count = 0

# Main loop
# while not done:
while True:

    # Preprocess observation
    obs = obs[0]

    ### INTERPRETABILITY
    with torch.no_grad():
        # We have to repeat part of the processing done by the search and the model
        # Transform to tensor
        if len(obs) == 2:
            obs_t = torch.tensor(obs[0], device=device)
        else:
        	obs_t = torch.tensor(obs, device=device)

        # First latent representation
        representation = nec_mu_net.represent(obs_t.unsqueeze(0))

        # Apply layers from the prediction net to get to the DND representation latent space
        latent = dnd_embedding(representation)

        # Predict outcome of using action 1 eight times (only the first iteration)
        if step_count == 0:
            print("PREDICTING THE OUTCOME OF MOVING 8 TIMES TO THE LEFT:")
            future_representation = torch.clone(representation)
            lstm_hiddens = (
                            torch.zeros(1, 1, config["lstm_hidden_size"]).detach(),
                            torch.zeros(1, 1, config["lstm_hidden_size"]).detach(),
                        )
            # ACTION = torch.tensor([0,1]).unsqueeze(0)  # One-hot encoded
            ACTION = torch.tensor([1,0]).unsqueeze(0)  # One-hot encoded
            TIMES = 8
            imgs = []
            for _ in range(TIMES):
                # Get prediction of the future latent state
                future_representation, reward, lstm_hiddens = nec_mu_net.dynamics(
                    future_representation, ACTION, lstm_hiddens
                )

                # Apply layers from the prediction net to get to the DND representation latent space
                future_latent = dnd_embedding(future_representation)

                # Append the raw observation (image) corresponding to the closest neighbor
                _, knn_indices = dnd.query_knn(future_latent, training=False)
                knn_indices = knn_indices[0]
                obs_pickle_path = os.path.join(RAW_OBSERVATIONS_PATH, str(knn_indices[0]))
                with open(obs_pickle_path,'rb') as obs_pickle:  
                    imgs.append( pickle.load(obs_pickle)[1] )

            # Show images
            f1 = show_images(imgs)
            f2 = plot_embedding_space(dnd, future_latent[0], knn_indices)
            f3 = plot_representations(dnd_elements_representations, future_representation[0], knn_indices)
            input("Press Enter to continue...")
            plt.close(f1)
            plt.close(f2)
            plt.close(f3)


        # Get outlier score
        outlier_score = -lof.decision_function(latent).reshape(-1,1)
        scaled_outlier_score = lof_scaler.transform( outlier_score )[0][0]
        print("Outlier Score:", outlier_score)
        print("Scaled Outlier Score:", scaled_outlier_score)
        print("Outlier Score Percentile:", percentileofscore(dnd_points_outlier_score, outlier_score[0]))

        # Get the neighbors
        dists, knn_indices = dnd.query_knn(latent, training=False)
        dists = dists[0]
        knn_indices = knn_indices[0]

        # Show similar cases in database
        imgs = []
        for i in range(NEIGHBORS_TO_DISPLAY):
            obs_pickle_path = os.path.join(RAW_OBSERVATIONS_PATH, str(knn_indices[i]))
            with open(obs_pickle_path,'rb') as obs_pickle:  
                imgs.append( pickle.load(obs_pickle)[1] )
        
        if step_count%20 == 0 or scaled_outlier_score > 2.0:
        # if True:
            f1 = show_images(imgs)
            f2 = plot_embedding_space(dnd, latent[0], knn_indices, 
                                      title=f"Paso {step_count} | LOF Normalizado: {scaled_outlier_score}")
            f3 = plot_representations(dnd_elements_representations, representation[0], knn_indices)
            input("Press Enter to continue...")
            plt.close(f1)
            plt.close(f2)
            plt.close(f3)

    ###

    # Choosing next action
    if ALWAYS_MOVE_LEFT: 
        obs, reward, terminated, truncated, _ = env.step(0, return_render=True)
        done = terminated or truncated
    else:
        tree = search(
            config, nec_mu_net, obs, minmax, device=device
        )
        print("(Debug) Prior action probability:", tree.pol_pred)
        action = tree.pick_game_action(temperature=0.0)
        print("Chosen action:", action)

        # Interaction and reading from the environment
        obs, reward, terminated, truncated, _ = env.step(action, return_render=True)
        done = terminated or truncated

    step_count += 1

    # Update total reward
    total_reward += reward

    # Render scene
    env.render()

env.close()

print("GAME OVER")
print("TOTAL REWARD:", total_reward)