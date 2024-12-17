import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.linear2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.linear3 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
