import torch
import numpy as np

class EarlyStopper:
    def __init__(self, patience: int = 5, minimize: bool = True, verb: bool = False):
        self.verb = verb
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
            if self.verb: print(f"New Best Model Performace and Epoch {self.epoch}")
            
            return False, True

        else:
            self.early_stop_counter += 1
            if self.verb: print(f"No improvement. Early stop counter = {self.early_stop_counter}")
            return self.early_stop_counter >= self.patience, False