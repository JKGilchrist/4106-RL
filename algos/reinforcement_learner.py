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
    def __init__(self, initial_threshold = 30, type_NN = 1, score_increase = 5):
        self.env = gym.make("CartPole-v0") #The game
        self.score_threshold = initial_threshold
        self.type_NN = type_NN
        self.model = False
        self.score_increase = score_increase
    

    '''
    The main class to run. Handles the playing and training iterations.
    All three parameters offer ways on how to indicate when the model should stop being trained.

    @param max_minutes is the maximum number of minutes to allow for a session of playing games to gather data to train on. If it's surpassed, then the model has finished being trained.  
    @param iter_cap is a hard max on the number of iterations it is allowed to train for. It may not be reached if max_minutes is reached first.
    @param max_decreases_in_a_row is a limit on how many iterations can the average game score decrease before training should stop.
    @param min_epochs is the minimum number of epochs that should occur. More (max of 100) may be performed.
    '''
    def train_repeatedly(self, max_minutes = 5, iter_cap = -1, max_decreases_in_a_row = 3, min_epochs = 30):

        #Used to track if training stops due to reaching max_decreases_in_a_row
        self.avg_game_scores = []
        stop = False 
        
        count = 1 # Used to track how many iterations are performed

        if iter_cap == -1: #AKA no iteration cap, only time cap
            while not stop: #Exits when max_minutes is reached or it decreases in avg score {max_decreases_in_a_row} times
                print("\nIteration #{}".format(count))
                attri, tar = self.play(100, max_minutes) #num good games to train on, if it can find that many in {max_minutes} minutes
                if attri == -1: #meaning max_minutes was reached
                    print("\nTime cap reached. Done training.")
                    break
                else:
                    epochs = -1
                    if len(self.avg_game_scores) > 1:
                        epochs = (self.avg_game_scores[-1] - self.avg_game_scores[-2]) * 2
                        epochs = min(epochs, 100) #Arbitrary limit to avoid overtraining, even if the average game score has dramatically increased.
                    epochs = max(min_epochs, epochs)
                    print("Performing {} number of epochs".format(epochs))
                    self.train_model(attri, tar, epochs)
                
                stop = self.check_for_decreases(max_decreases_in_a_row)
                if stop:
                    break
                
                count += 1
                
        else:
            for _ in range(iter_cap):
                print("\nIteration #{}".format(count))
                
                attri, tar = self.play(100, max_minutes) #num good games to train on, if it can find that many in {max_minutes} minutes
                
                if attri == -1: #Means 
                    print("\nTime cap reached. Done training.")
                    break

                epochs = -1
                if len(self.avg_game_scores) > 1:
                    epochs = self.avg_game_scores[-1] - self.avg_game_scores[-2]
                    epochs = min(epochs, 100) #Arbitrary limit to avoid overtraining, even if the average game score has dramatically increased.
                epochs = max(min_epochs, epochs)
                print("Performing {} number of epochs".format(epochs))
                self.train_model(attri, tar, epochs)
                
                stop = self.check_for_decreases(max_decreases_in_a_row)
                if stop:
                    break

                count += 1

            if count - 1 == iter_cap:
                count -=1 #It adds an extra
                print("\nIteration cap reached. Done training.")

        if stop:
            print("\nMax decreases in a row reached. Done training.")

        #Finished training results
        print("\n~~~~~~~~~~~~~~~\nResults\n~~~~~~~~~~~~~~~\n")
        print("Took {} iterations to get the following results.".format(count))
        self.score_threshold = 0
        self.final_results()
    

    '''
    Helper function to see if the model has decreased in average game scores, max_decreases number of times.

    @param max_decreases is the max number of decreases in a row.
    '''
    def check_for_decreases(self, max_decreases):
        if len(self.avg_game_scores) > max_decreases:
            lst = [] #Stores in reverse order the last {max_decreases + 1} entries of self.avg_game_scores

            for i in range(-1, -2 - max_decreases , -1):
                lst.append(self.avg_game_scores[i])
            
            if lst == sorted(lst):
                return True

        return False

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
            self.env._max_episode_steps = 500
            prev_observation = []
            game_memory = []
            score = 0
            num_games += 1

            while True: #Plays until the game is done

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
            return -1, -1

        else:
            print("Finished playing: {}".format(str(datetime.datetime.now())))
            stats = training_data.get_stats()
          
            self.avg_game_scores.append( stats["avg game score"] )

            print("Played {total} games to get {good_total} games with a score >= {thres}. Avg game score: {avg1}, Avg good game score: {avg2}".format(total = stats["total games"], good_total = stats["total good games"], avg1 = round(stats["avg game score"], 3), avg2 = round(stats["avg good game score"], 3), thres = self.score_threshold))
            
            self.score_threshold += self.score_increase
            
            if stats["avg game score"] - 10 > self.score_threshold: #Force it to play better games
                self.score_threshold = stats["avg game score"] + self.score_increase
            
            return training_data.get_training_data()
    
    
    '''
    Gathers stats on model for 100 games.
    '''
    def final_results(self):
        training_data = Training_Data_Collector(self.score_threshold, 0)
        
        #play 100 games
        for _ in range(100):

            self.env.reset()
            self.env._max_episode_steps = 500

            prev_observation = []
            game_memory = []
            score = 0
            
            while True: #Plays until the game is done

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
        
        stats = training_data.get_stats()
        print("In 100 games, the avg game score this model achieves is: {avg1}".format(avg1 = round(stats["avg game score"], 3) ))


    '''
    Creates model if it doesn't exist. 
    Trains it.

    @param attributes is a list of lists, each containing the values of the observation.
    @param targets is a list of desired movements for the corresponding observation. 0 is left, 1 is right
    '''
    def train_model(self, attributes, targets, epochs):
        if not self.model:
            if self.type_NN == 1:
                self.model = logistic_regression()
            if self.type_NN == 2:
                self.model = None #TODO ...
            if self.type_NN == 3:
                self.model = None #TODO

        self.model.train(attributes, targets, epochs)

    '''
    Renders a single game. Not involved in gathering data/training process
    '''
    def view_a_game(self):
        start_time = datetime.datetime.now()   
        print("\n~~~~~~~~~~~~~~~\nPlaying and rendering a single game\n")     
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
    Closes the environment. Only really necessary if view_a_game is run (to close the render window)
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
        print("Creating new model, {}".format(filename))
        
        rl = reinforcementLearner(initial_threshold = 30, type_NN = int(sys.argv[1]), score_increase = 20) 
        
    #View an initial game
    #rl.view_a_game()

    #Train!
    rl.train_repeatedly(max_minutes = 1.5, iter_cap = 10, max_decreases_in_a_row = 3, min_epochs = 30) 
    
    #save the model
    pickle.dump(rl, open(filename, "wb"  ) )
    print("\nSaved trained model, {}".format(filename))

    #watch it play
    rl.view_a_game()
    rl.close()
