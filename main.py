import torch
import os
import random
from datetime import datetime
import numpy as np

from agents.dqn_agent import DQNAgent
from env.snake_environment import SnakeEnvironment
from config.hyperparameters import Config
from utils.plotter import Plotter
from utils.logger import Logger

def main():
    # Load configuration
    config = Config()
    seed = config.get('general', 'seed')
    params = config.get_section('training')

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up directories
    run_number = 1
    base_save_dir = 'saved_models'
    while os.path.exists(os.path.join(base_save_dir, f'run_{run_number}')):
        run_number += 1
    run_dir = os.path.join(base_save_dir, f'run_{run_number}')
    os.makedirs(run_dir)

    # Save hyperparameters.json to run directory
    config.save(os.path.join(run_dir, 'hyperparameters.json'))

    # Initialize environment and agent
    game_env = SnakeEnvironment()
    agent = DQNAgent(game_env, config, run_dir)

    # Initialize plotter and logger
    plotter = Plotter(run_dir)
    logger = Logger(run_dir)

    best_score = 0

    for episode in range(1, params['num_episodes'] + 1):
        state = game_env.reset()
        done = False
        current_score = 0

        while not done:
            action = agent.get_action(state)
            reward, done, score = game_env.step(action)
            next_state = game_env.get_state()

            # Train short-term memory
            agent.train_short_memory(state, action, reward, next_state, done)

            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            current_score = score

        # Train long-term memory after each episode
        agent.train_long_memory()

        # Increment game count
        agent.n_games += 1

        # Save model if it's the best so far
        if current_score > best_score:
            best_score = current_score
            agent.save_model(os.path.join(run_dir, 'model.pth'))

        # Update plots and logs
        plotter.plot(current_score, episode)
        logger.log(episode, current_score, best_score)

        print(f'Episode: {episode}, Score: {current_score}, Record: {best_score}')

    game_env.close()

if __name__ == "__main__":
    main()