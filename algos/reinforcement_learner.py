import gym

import datetime
import time
import pickle
import sys

from log_reg import logistic_regression
from support import Training_Data_Collector
#TODO add your imports


'''
The reinforcement learner class, handling it all
'''
class reinforcementLearner:


    '''
    Constructor.

    @param initial_threshold establishes the first minimum score for good games, that will be used to train
    @param type_NN indicates which NN to use (1 is logistic regression, 2 is TODO fill in, 3 is TODO fill in )
    @param score_increase indicates the minimum amount of how much the threshold should increase by with each iteration
    '''
    def __init__(self, initial_threshold, type_NN, score_increase):
        self.env = gym.make("CartPole-v0") #The game
        self.score_threshold = initial_threshold
        self.type_NN = type_NN
        self.model = False
        self.score_increase = score_increase
    

    '''
    The main class to run. Handles the playing and training iterations.

    @param max_minutes is the maximum number of minutes to allow for a session of playing games to gather data to train on. If it's surpassed, then the model has finished being trained.  
    '''
    def train_repeatedly(self, max_minutes):
        #TODO (optional) add bit that tracks performance over series of iterations. If avg game score decreases over x number of iterations, stop training. 

        count = 1
        try:
            while True:
                print("\nIteration #{}".format(count))
                attri, tar = self.play(100, max_minutes) #num good games to train on, if it can find that many in 3 minutes
                self.train_model(attri, tar)
                count += 1
        except:
            print("\n~~~~~~~~~~~~~~~")
            print("Results")
            print("~~~~~~~~~~~~~~~\n")
            print("Took {} iterations to get the following results.".format(count))
            self.score_threshold = 0
            self.play(1 , 5) #Just plays 100 games. Time cap should be irrelevant


    '''
    Gathers data by playing the game repeatedly.

    @param target_good_games is the number of games expected in the returned training data
    @param time_cap is the maximum number of minutes to be spent playing games. If it surpasses, then an exception is thrown
    '''
    def play(self, target_good_games, time_cap):
        training_data = Training_Data_Collector(self.score_threshold, target_good_games)
        num_games = 0
        print("Started playing: {}".format(str(datetime.datetime.now())))
        cap_time = time.time() + 60 * time_cap #Cap on how long RL can play, looking for good games. 

        cut_short = False
        # a series of games
        while not training_data.at_target():

            play_time = time.time()
            if play_time > cap_time:
                print("Exceeded data-gathering time limit. Training complete.")
                cut_short = True
                break
            
            self.env.reset()
            prev_observation = []
            game_memory = []
            score = 0
            num_games += 1

            while True: #Plays until the game is done
                #self.env.render() #If you need to display the game

                if len(prev_observation) and self.model: #If there's a prev observation and the model exists 
                    action = self.model.predict(prev_observation)[0]
                else:
                    action = self.env.action_space.sample() #Selects a random action
                
                observation, reward, done, _info = self.env.step( action ) #Performs the action
                
                if len(prev_observation):
                    game_memory.append([prev_observation, action])

                prev_observation = observation
                
                score += reward
                
                if done:
                    break

            training_data.add_game(game_memory, score)
        
        if cut_short:
            raise Exception("Done training.") 

        else:
            if target_good_games == 1: #Indicates gathering stats for end results, not gathering training data
                print("Finished playing: {}".format(str(datetime.datetime.now())))
                stats = training_data.get_stats()
                print("In {total} games, the avg game score this model achieves is: {avg1}".format(total = stats["total games"], avg1 = stats["avg game score"] ))

            else: #The standard ending when gathering stats. Informs programmer and returns the training data for model's training
                print("Finished playing: {}".format(str(datetime.datetime.now())))
                stats = training_data.get_stats()
                print("Played {total} games to get {good_total} games with a score >= {thres}. Avg game score: {avg1}, Avg good game score: {avg2}".format(total = stats["total games"], good_total = stats["total good games"], avg1 = stats["avg game score"], avg2 = stats["avg good game score"], thres = self.score_threshold))
                
                self.score_threshold += self.score_increase
                
                if stats["avg game score"] - 10 > self.score_threshold: #Force it to play better games
                    self.score_threshold = stats["avg game score"] + self.score_increase
                
                return training_data.get_training_data()
    
    '''
    Creates model if it doesn't exist. 
    Trains it.

    @param attributes is a list of lists, each containing the values of the observation.
    @param targets is a list of desired movements for the corresponding observation. 0 is left, 1 is right
    '''
    def train_model(self, attributes, targets):
        print(targets)
        if not self.model:
            if self.type_NN == 1:
                self.model = logistic_regression()
            if self.type_NN == 2:
                self.model = None #TODO ...
            if self.type_NN == 3:
                self.model = None #TODO

        self.model.train(attributes, targets)


    '''
    Renders a single game. Not involved in gathering data/training process
    '''
    def view_a_game(self):
        start_time = datetime.datetime.now()   
        print("\n\n~~~~~~~~~~~~~~~")
        print("Playing and rendering a single game\n")     
        print("Start time:", start_time)
        self.env.reset()
        prev_observation = []

        score = 0
        while True: #Plays until the game is done
            self.env.render() 

            if len(prev_observation) and self.model: #If there's a prev observation and the model exists 
                action = self.model.predict(prev_observation)[0]
            else:
                action = self.env.action_space.sample() #Selects a random action
                
            observation, reward, done, _info = self.env.step( action ) #Performs said action
                
            prev_observation = observation
                
            score += reward
                
            if done:
                break
        end_time = datetime.datetime.now()
        print("End time: {}".format(str(end_time)))
        print("Total time played: {}".format(end_time - start_time))
        print("score:", score)
        

    '''
    Closes the environment. Only really necessary if view_a_game is run (as it closes the render window)
    '''
    def close(self):
        self.env.close()
        


if __name__ == "__main__":

    #Determines which NN to use
    if int(sys.argv[1]) == 1:
        filename = "rl_lr.obj"
    elif int(sys.argv[1]) == 2:
        filename = "rl_cnn.obj"
    elif int(sys.argv[1]) == 3:
        filename = "rl_rnn.obj"
    else:
        raise Exception("Indicate which NN to use")

    #Tries to open previous file
    try:
        filehandler = open(filename, 'rb')
        rl = pickle.load( filehandler )
        print("Loaded saved model, {}".format(filename))
    except:
        print("Creating new model")
        rl = reinforcementLearner(50, int(sys.argv[1]), 20) #initial score threshold, NN type, score threshold increase amount
    
    #Train!
    rl.train_repeatedly(0.5) #Max time spent in one interation gathering data

    #save the model
    pickle.dump(rl, open(filename, "wb"  ) )

    #watch it play
    rl.view_a_game()
    rl.close()
    