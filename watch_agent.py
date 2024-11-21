import os
import torch
import imageio
import numpy as np
import pygame  # Added this import
from agents.dqn_agent import DQNAgent
from env.snake_environment import SnakeEnvironment
from config.hyperparameters import Config
import matplotlib.pyplot as plt  # For frame verification

def watch_trained_agent():
    # Load configuration
    config = Config()
    saved_models_dir = 'saved_models'

    # List available runs and sort them numerically
    available_runs = [d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))]
    available_runs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by the numerical part after 'run_'
    if not available_runs:
        print("No saved runs found in 'saved_models' directory.")
        return

    print("Available runs:")
    for run in available_runs:
        print(f"- {run}")

    run_number = input('Enter the run number to watch (e.g., 1 for run_1): ')
    run_dir = os.path.join(saved_models_dir, f'run_{run_number}')

    if not os.path.exists(run_dir):
        print(f"Run '{run_dir}' does not exist.")
        return

    # Initialize environment and agent
    game_env = SnakeEnvironment()
    agent = DQNAgent(game_env, config, run_dir)

    seed = config.get('general', 'seed')
    game_env = SnakeEnvironment(seed=seed)  # Pass seed to environment
    agent = DQNAgent(game_env, config, run_dir)

    # Load the trained model
    model_path = os.path.join(run_dir, 'model.pth')
    if not os.path.exists(model_path):
        print(f"Model file not found at '{model_path}'.")
        return
    agent.model.load(model_path)

    # Set agent to evaluation mode
    agent.training = False
    agent.model.eval()

    # Number of episodes to play
    num_episodes = 5

    all_episode_frames = []
    all_episode_scores = []

    for episode_num in range(num_episodes):
        print(f"Playing episode {episode_num+1}/{num_episodes}")

        # Prepare to capture frames for this episode
        frames = []

        # Reset the environment for a new episode
        state = game_env.reset()
        done = False
        total_score = 0

        while not done:
            # Handle Pygame events to prevent the window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            action = agent.get_action(state)
            reward, done, score = game_env.step(action)
            state = game_env.get_state()
            total_score = score

            # Render the game and capture the frame
            game_env.render()
            # Capture the frame from the Pygame display
            frame = game_env.get_frame()
            if frame is not None:
                frames.append(frame)
            else:
                print("Warning: Captured frame is None.")

            # Optional: Add a short delay to control the speed of the game during playback
            pygame.time.delay(50)

        print(f'Episode {episode_num+1} - Score: {total_score}')

        # Store the frames and score of this episode
        all_episode_frames.append(frames)
        all_episode_scores.append(total_score)

    # Close the game environment
    game_env.close()

    # Save the first episode as a GIF
    first_episode_frames = all_episode_frames[0]
    first_episode_gif_path = os.path.abspath(os.path.join(run_dir, 'first_episode.gif'))
    imageio.mimsave(first_episode_gif_path, first_episode_frames, fps=10)
    print(f'First episode GIF saved at {first_episode_gif_path}')

    # Find the episode with the highest score
    max_score = max(all_episode_scores)
    best_episode_index = all_episode_scores.index(max_score)
    best_episode_frames = all_episode_frames[best_episode_index]
    best_episode_gif_path = os.path.abspath(os.path.join(run_dir, 'best_episode.gif'))
    imageio.mimsave(best_episode_gif_path, best_episode_frames, fps=10)
    print(f'Best episode GIF saved at {best_episode_gif_path} with score {max_score}')

if __name__ == '__main__':
    watch_trained_agent()