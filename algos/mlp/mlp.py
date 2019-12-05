import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import random

class multi_layer_perceptron:
    
    def __init__ multi_layer_perceptron(self):
        model = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(150, 150), random_state=1, max_iter=100, learning_rate_init=0.01, warm_start=True)

    def train(self, df_attributes, df_target, epochs=50):
        while i < range(1, epochs + 1):
            self.fit(df_attributes, df_target)
            i+=1

    def predict(self, prev_state):
        scores = []
        for x in prev_state:
            score = self.score(x)
            scores.append(score)
        return scores
        