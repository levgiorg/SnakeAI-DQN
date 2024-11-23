import csv
import os

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, 'training_log.csv')
        with open(self.filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Score', 'Record'])

    def log(self, episode, score, record):
        with open(self.filepath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([episode, score, record])