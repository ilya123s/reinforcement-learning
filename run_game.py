import os
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kaggle_environments import make
from stable_baselines import PPO1
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

# Create directory for logging training information
from ConnectFourGym import ConnectFourGym
from create_model import get_model
from utils import get_win_percentages

model = PPO1


def agent1(obs, config):
    # Use the best model to select a column
    global model
    col, _ = model.predict(np.array(obs['board']).reshape(6, 7, 1))
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move.
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])


def train_model(model):
    model.learn(total_timesteps=1000000)

    # Plot cumulative reward
    with open(os.path.join(log_dir, "monitor.csv"), 'rt') as fh:
        firstline = fh.readline()
        assert firstline[0] == '#'
        df = pd.read_csv(fh, index_col=None)['r']
    df.rolling(window=1000).mean().plot()
    plt.show()
    return model


if __name__ == '__main__':
    env = ConnectFourGym(agent2="random")
    log_dir = "ppo/"
    os.makedirs(log_dir, exist_ok=True)

    # Logging progress
    monitor_env = Monitor(env, log_dir, allow_early_resets=True)

    # Create a vectorized environment
    vec_env = DummyVecEnv([lambda: monitor_env])

    # Initialize agent
    model = get_model(vec_env)

    # Train agent
    model = train_model(model)

    env_game = make("connectx")
    env_game.run([agent1, "random"])
    get_win_percentages(agent1=agent1, agent2="random")
