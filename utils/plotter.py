import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.scores = []
        self.mean_scores = []
        self.total_score = 0

    def plot(self, score, episode):
        self.scores.append(score)
        self.total_score += score
        mean_score = self.total_score / episode
        self.mean_scores.append(mean_score)

        plt.figure(figsize=(10,5))
        plt.clf()
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(self.scores, label='Score')
        plt.plot(self.mean_scores, label='Average Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'training_plot.png'))
        plt.close()