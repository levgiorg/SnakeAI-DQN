import torch
import random
import numpy as np
from collections import deque

from models.neural_net import NeuralNetwork
from models.q_trainer import QTrainer
from utils.replay_memory import ReplayMemory

class DQNAgent:
    def __init__(self, environment, config, save_dir):
        self.n_games = 0
        self.epsilon = config.get('agent', 'epsilon_start')  # Initialize epsilon
        self.epsilon_min = config.get('agent', 'epsilon_min')  # Minimum epsilon
        self.gamma = config.get('agent', 'gamma')  # Discount factor
        self.memory_capacity = config.get('agent', 'memory_size')
        self.epsilon_decay = config.get('agent', 'epsilon_decay')
        self.batch_size = config.get('agent', 'batch_size')
        self.memory = ReplayMemory(self.memory_capacity)
        self.config = config
        self.save_dir = save_dir
        self.training = True  # Add training flag

        # Model and trainer
        input_size = environment.state_size
        hidden_size = config.get('agent', 'hidden_size')
        output_size = environment.action_space.n
        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        self.trainer = QTrainer(self.model, lr=config.get('agent', 'learning_rate'), gamma=self.gamma)

    def get_action(self, state):
        final_move = [0, 0, 0]
        
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if random.random() < self.epsilon:
                # Random action
                move = random.randint(0, 2)
                print(f"Epsilon: {self.epsilon}, n_games: {self.n_games}")
                print(f"Random action selected: {move}")
            else:
                # Predicted action
                state0 = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.trainer.device)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                print(f"Epsilon: {self.epsilon}, n_games: {self.n_games}")
                print(f"Predicted action selected: {move}")
        else:
            # Evaluation mode: always select the best action
            state0 = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.trainer.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            print(f"Predicted action selected: {move}")
        
        final_move[move] = 1
        return move

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_batch = self.memory.sample(self.batch_size)
        else:
            mini_batch = self.memory.sample(len(self.memory))

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def save_model(self, file_path):
        self.model.save(file_path)
