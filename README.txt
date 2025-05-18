## Overview
This folder contains MATLAB scripts and supporting files for training and testing reinforcement learning (RL) algorithms on a pendulum environment. The provided scripts implement a policy gradient method for controlling the pendulum.

## Files and Directories

### `PendulumEnv.m`
This script defines the pendulum environment for the RL algorithm. It includes the dynamics and reward function for the pendulum.

### `train_env_PG.m`
This script is used to train the RL agent using the policy gradient method. It initializes the environment, sets up the RL agent, and performs the training loop.

### `test_PG.m`
This script tests the trained RL agent in the pendulum environment. It loads the trained agent and evaluates its performance.

## Usage

1. **Training the RL Agent:**
   - Run `train_env_PG.m` to train the RL agent. This script will use the pendulum environment defined in `PendulumEnv.m` and save the trained agent in the `PG/` directory.

2. **Testing the RL Agent:**
   - Run `test_PG.m` to test the performance of the trained RL agent. This script will load the trained agent from the `PG/` directory and evaluate its performance in the pendulum environment.

## Notes
- Make sure all the necessary files are in the same directory or update the file paths accordingly in the scripts.
- The `PG/` directory contains pre-trained models and visualizations that can be used for reference or further analysis.
