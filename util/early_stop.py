import torch
import numpy as np

class EarlyStopper:
    def __init__(self, patience: int = 5, minimize=True):
        self.best_score = None
        self.epoch = 0
        self.early_stop_counter = 0
        self.patience = patience
        self.comp = lambda old, new: new < old if minimize else lambda old, new: new > old

    def update(self, score) -> bool:
        """
        Update best score tracking with new score if new best
        Returns a bool indicating whether to early stop or not
        """
        self.epoch += 1

        if self.best_score is None or self.comp(score, self.best_score):
            self.best_score = score
            self.early_stop_counter = 0
            print(f"New Best Model Performace and Epoch {self.epoch}")
            
            return False

        else:
            self.early_stop_counter += 1
            print(f"No improvement. Early stop counter = {self.early_stop_counter}")
            return self.early_stop_counter >= self.patience