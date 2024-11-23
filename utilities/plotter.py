import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.scores = []
        self.mean_scores = []
        self.total_score = 0

    def add_score(self, score):
        """Add a new score and update the mean score."""
        self.scores.append(score)
        self.total_score += score
        mean_score = self.total_score / len(self.scores)
        self.mean_scores.append(mean_score)

    def save_plot(self, filename='training_plot.png'):
        """Generate and save the training plot."""
        plt.figure(figsize=(10, 5))
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(self.scores, label='Score')
        plt.plot(self.mean_scores, label='Average Score')
        plt.legend()
        plt.grid(True)
        
        # Ensure the save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        save_path = os.path.join(self.save_dir, filename)
        try:
            plt.savefig(save_path)
            print(f"Training plot saved successfully at {save_path}")
        except OSError as e:
            print(f"Error saving training plot: {e}")
        finally:
            plt.close()
