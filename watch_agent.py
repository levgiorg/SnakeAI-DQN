import os
import torch
import imageio
import numpy as np
import pygame
import random

from agents import DQNAgent
from environment import SnakeEnvironment
from config import Config

def watch_trained_agent():
    # Initialize Pygame
    pygame.init()

    # Load configuration
    config = Config()
    runs = 'runs'

    # List available runs and sort them numerically
    available_runs = [d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d))]
    available_runs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by the numerical part after 'run_'
    if not available_runs:
        print("No saved runs found in 'runs' directory.")
        pygame.quit()
        return

    print("Available runs:")
    for run in available_runs:
        print(f"- {run}")

    run_number = input('Enter the run number to watch (e.g., 1 for run_1): ').strip()
    run_dir = os.path.join(runs, f'run_{run_number}')

    if not os.path.exists(run_dir):
        print(f"Run '{run_dir}' does not exist.")
        pygame.quit()
        return

    # Set seeds for reproducibility
    seed = config.get('general', 'seed', default=42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize environment and agent
    game_env = SnakeEnvironment(seed=seed)
    agent = DQNAgent(game_env, config, run_dir)

    # Load the trained model
    model_path = os.path.join(run_dir, 'model.pth')
    if not os.path.exists(model_path):
        print(f"Model file not found at '{model_path}'.")
        game_env.close()
        pygame.quit()
        return
    try:
        agent.model.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        game_env.close()
        pygame.quit()
        return
    agent.model.eval()
    agent.training = False

    # Define parameters
    total_episodes = 100  
    best_score = -float('inf')
    best_episode_frames = []
    best_episode_num = -1
    FPS = 120  # High FPS for faster gameplay
    clock = pygame.time.Clock()

    print(f"\n--- Running {total_episodes} Episodes to Find the Best Score ---")
    for episode in range(1, total_episodes + 1):
        print(f"\nStarting Episode {episode}/{total_episodes}")
        state = game_env.reset()
        done = False
        current_score = 0
        frames = []  # Capture frames for the current episode

        while not done:
            # Handle Pygame events to prevent the window from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Pygame window closed by user.")
                    game_env.close()
                    pygame.quit()
                    return

            action = agent.get_action(state)
            reward, done, score = game_env.step(action)
            state = game_env.get_state()
            current_score = score

            # Render the game and capture the frame
            game_env.render()
            frame = game_env.get_frame()
            if frame is not None:
                frames.append(frame)

            # Control the speed of the game during playback
            clock.tick(FPS)

        print(f'Episode {episode} - Score: {current_score}')

        if current_score > best_score:
            best_score = current_score
            best_episode_num = episode
            best_episode_frames = frames.copy()  # Save frames of the best episode
            print(f"New Best Score: {best_score} at Episode {best_episode_num}")

    print(f"\n--- Completed {total_episodes} Episodes ---")
    if best_episode_num != -1:
        print(f"Best Score Achieved: {best_score} at Episode {best_episode_num}")
    else:
        print("No episodes were completed successfully.")
        game_env.close()
        pygame.quit()
        return

    # Save the GIF of the best episode
    best_episode_gif_path = os.path.abspath(os.path.join(run_dir, 'best_episode.gif'))
    try:
        imageio.mimsave(best_episode_gif_path, best_episode_frames, fps=10)
        print(f'Best episode GIF saved at {best_episode_gif_path} with score {best_score}')
    except Exception as e:
        print(f"Error saving GIF: {e}")

    # Close the game environment and quit Pygame
    game_env.close()
    pygame.quit()

if __name__ == '__main__':
    watch_trained_agent()
