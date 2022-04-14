# Double DQN with Replay Buffer (no prioritization)
import gym
import random
import numpy as np
import wandb
from tqdm import trange
import sys, yaml
import torch
import torch.optim as optim
from models import ReplayBuffer, Qnet
from utils import train, get_moving_average

from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_episodes(config):

    env = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = Qnet() .to(device)             # Initialize Behavior Network
    q_target = Qnet().to(device)     # Initialize Target Network
    q_target.load_state_dict(q.state_dict())    # Copy weights of q to q_target

    memory = ReplayBuffer(config)

    score = 0.0
    prev_score = 0.0
    prev_ep = 1
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)   # only optimize q (since we'll copy weights to target later)
    episode_durations = []
    moving_avrg_period = 100
    moving_average = 0.0

    for n_epi in range(1, max_episode+1):
        epsilon = max(0.01, 0.1 - 0.01*(n_epi/200))     # Update epsilon (reduce by 1% every 200 episodes)
        s = env.reset()                                 # reset and get first state
        done = False                                    # Done -> True when episode ends
        print_interval = 20
        while not done:

            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)
            # "s" in tensor format: torch.from_numpy(s).float() = tensor([0.0173, 0.0048, -0.0356, -0.0088])
            s_prime, r, done, info = env.step(a)

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))     # to avoid over-fitting by large r value
            s = s_prime

            score += r
            # if done:
        moving_average = get_moving_average(episode_durations, moving_avrg_period)    #    break
        if n_epi < moving_avrg_period:
            episode_durations.append(score)
        else:
            episode_durations.pop(0)
            episode_durations.append(score)

        if score > prev_score:  # stores max score and episode when it happens
            prev_ep = n_epi
            prev_score = score

        wandb.log({"score": score, "max_avg_scr": prev_score, "maxscr_ep": prev_ep, "moving_average": moving_average})  # logs episode score to wandb for comparizon
        tune.report(score=moving_average)  # optimize on which gets the best reward faster

        if memory.size() > config["initial_exp"]:     # Training after storing some data(> initial_exp)
            train(q, q_target, memory, optimizer, config["batch_size"], device, config["gamma"])

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())  # Update q_target weights copied from q every c episodes
            str_updte = "n_ep : {}, scr : {:.1f}, max_scr : {:.1f}, max_scr_ep : {}, eps : {:.1f}%".format(
                n_epi, score, prev_score, prev_ep, epsilon * 100)
            # print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            #                                     n_epi, score/print_interval, memory.size(), epsilon*100))
            print(str_updte)  # to show immediately the update

        score = 0.0

    env.close()


if __name__ == '__main__':
    cnfg_path = "./config.yml"
    cnfg = open(cnfg_path, 'r')
    config_dict = yaml.load(cnfg, Loader=yaml.FullLoader)

    # Hyper-parameters
    learning_rate = config_dict['lr']
    gamma = config_dict['gamma']
    max_episode = config_dict['max_episode'] # maybe same as buffer
    buffer_limit = config_dict['buffer_limit']    # buffer max size
    batch_size = config_dict['batch_size']        # TBD
    initial_exp = config_dict['initial_exp']      # TBD

    # W&B run
    WB_API_KEY = config_dict['WB_API_KEY']
    project = config_dict['project']
    metric = config_dict['metric']
    mode = config_dict['mode']
    num_samples = config_dict['num_samples']

    analysis = tune.run(
        train_episodes,
        config={
            "batch_size": tune.grid_search(batch_size),
            "initial_exp": tune.grid_search(initial_exp),
            "gamma": tune.grid_search(gamma),
            "buffer_limit": tune.grid_search(buffer_limit),
            "wandb": {
                "project": project,
                "api_key": WB_API_KEY
            }
        },
        metric=metric,
        mode=mode,
        num_samples=num_samples)
