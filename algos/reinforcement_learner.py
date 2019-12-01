import gym
from log_reg import logistic_regression

from support import Training_Data_Collector

import datetime
import pickle
import sys

class reinforcementLearner:

    def __init__(self, initial_threshold, type_NN, score_increase):
        self.env = gym.make("CartPole-v0")
        self.score_threshold = initial_threshold
        self.type_NN = type_NN
        self.model = False
        self.score_increase = score_increase
    
    def play(self, target_good_games = 30):
        training_data = Training_Data_Collector(self.score_threshold, target_good_games)
        num_games = 0
        print("Started playing: {}".format(str(datetime.datetime.now())))

        # a series of games
        while not training_data.at_target():
            self.env.reset()
            prev_observation = []
            game_memory = []
            score = 0
            num_games += 1

            #moves of the game
            for _ in range(10000): #Unlikely that any game will reach that high
                #self.env.render() #If you need to display the game

                if len(prev_observation) and self.model: #The first move is always random
                    action = self.model.predict(prev_observation)[0]
                    
                else:
                    action = self.env.action_space.sample() #Selects a random actions
                
                observation, reward, done, _info = self.env.step( action ) #Performs it 
                
                if len(prev_observation):
                    game_memory.append([prev_observation, action])
                prev_observation = observation
                score += reward
                if done:
                    break

            training_data.add_game(game_memory, score)
        
        print("Finished playing: {}".format(str(datetime.datetime.now())))
        stats = training_data.get_stats()
        print("Played {total} games to get {good_total} good games. Avg game score: {avg1}, Avg good game score: {avg2}".format(total = stats["total games"], good_total = stats["total good games"], avg1 = stats["avg game score"], avg2 = stats["avg good game score"]))
        self.score_threshold += self.score_increase
        return training_data.get_training_data()
    
    def train_model(self, attributes, targets):
        if not self.model:
            if self.type_NN == 1:
                self.model = logistic_regression()

        self.model.train(attributes, targets)

    #TODO add a def view a game


if __name__ == "__main__":

    if int(sys.argv[1]) == 1:
        filename = "rl_lr.obj"
    elif int(sys.argv[1]) == 2:
        filename = "rl_cnn.obj"
    elif int(sys.argv[1]) == 3:
        filename = "rl_rnn.obj"
    else:
        raise Exception("Indicate which NN to use")

    try:
        filehandler = open(filename, 'rb')
        rl = pickle.load( filehandler )
        print("loaded!")
    except:
        rl = reinforcementLearner(15, int(sys.argc[1]), 5) #initial score threshold, NN type, score threshold increase amount
    
    iterations = 10

    for x in range(iterations):
        print("iteration:", x + 1)
        attri, tar = rl.play(100) #num games, 
        rl.train_model(attri, tar)
    pickle.dump(rl, open('rl_lr.obj', "wb"  ) )