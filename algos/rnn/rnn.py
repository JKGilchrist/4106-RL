
#ref: https://www.tensorflow.org/guide/keras/rnn
#ref: https://keras.io/getting-started/sequential-model-guide/
#ref: https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import collections
import tensorflow as tf
from tensorflow.keras import layers as klayers

'''
Class handling the recurrent neural net model
'''
class recurrent_neural_net:

    '''
    Constructor. Creates the model.
    '''
    def __init__(self):
        self.model = tf.keras.Sequential() 
        #Dropout selects nodes to ignore during training to avoid over-fitting
        self.model.add(klayers.Dropout(0.5))
        #LAYERS

        #Experimenting with different activation & layer types.
        #Long Short-Term memory layers require 3D data set, so must reshape 1D arrays using the 'input_shape' parameter.
            #Reshaped data seems ineffective? Removing all LSTM layers in favour of Dense layers
        #self.model.add(LSTM(50), input_shape = (1,10,2))
        self.model.add(klayers.Dense(10, activation='sigmoid'))
        self.model.add(klayers.Dense(20, activation='relu'))
        self.model.add(klayers.Dense(2, activation='linear'))
        #COMPILE
        #Requirig the RNN to compile improves efficiency. It converst the layers into a concise matrix.
        #Here is where loss function and optimization algorithm is defined.
        self.model.compile(loss='mse', optimizer='sgd')

    '''
    Trains the model the given number of epochs.

    @param df_attributes is a list of 1D numpy arrays containing the observations
    @param df_target is a 1D list of the desired output or move, with respect to the observations
    @param epochs is the number of epochs it should perform with this data
    '''
    def train(self, df_attributes, df_target, epochs=100): 
        print("Started training: {}".format(str(datetime.datetime.now())))
        for i in range(int(epochs)):
            self.model.fit(np.asarray(df_attributes), np.asarray(df_target))
        print("Finished training: {}".format(str(datetime.datetime.now())))
    

    '''
    Given the state of the board, what move should be made.

    @param prev_state is a 1D numpy array containing the current game state.
    '''
    def predict(self, prev_state):
        
        #Has to convert to a normal array to perform the predict on it
        lst = []
        for x in prev_state:
            lst.append(x)

        predicted = self.model.predict([lst])
        
        return predicted