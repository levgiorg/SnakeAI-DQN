import torch
import torch.nn as nn
import torch.optim as optim

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to tensors if they aren't already
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        if len(state.shape) == 1:
            # If we're training with a single sample, add a batch dimension
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Predicted Q values with current state
        pred = self.model(state)

        # Target Q values
        target = pred.clone().detach()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx]] = Q_new

        # Zero gradients
        self.optimizer.zero_grad()
        # Compute loss
        loss = self.criterion(pred, target)
        # Backpropagation
        loss.backward()
        # Update weights
        self.optimizer.step()