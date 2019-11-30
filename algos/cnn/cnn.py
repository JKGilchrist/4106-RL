import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import random

class mlp:
    
    def __init__ mlp(solver, alpha, hiddenLayer1, hiddenLayer2, randState, maxIter, learningRate, warmStart)
        model = MLPClassifier(solver, alpha, hiddenLayer1, hiddenLayer2, randState, maxTter, learningRate, warmSart)

    def trainNEval(model, x_train, y_train, x_test, y_test, epochs):
        trainScores = []
        testScores = []
        i = 0
        while i < range(1, epochs + 1):
            clf.fit(x_train, y_train)
            trainScore = clf.score(x_train, y_train)
            trainScores.append(trainScore)
            testScore = clf.score(x_test, y_test)
            testScores.append(testScore)
            i+=1
        return trainScores, testScores

        def ohe(seed):
            random.seed(seed)
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        return ohe