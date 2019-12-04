#Ref Notebook 4

from sklearn.linear_model import LogisticRegression
import datetime

'''
Class handling the logistic regression model
'''
class logistic_regression:

    '''
    Constructor. Creates the model.
    '''
    def __init__(self):
        self.clf = LogisticRegression(solver='lbfgs', multi_class="multinomial", max_iter=1000, random_state=1)

    '''
    Trains the model the given number of epochs.

    @param df_attributes is a list of 1D numpy arrays containing the observations
    @param df_target is a 1D list of the desired output or move, with respect to the observations
    @param epochs is the number of epochs it should perform with this data
    '''
    def train(self, df_attributes, df_target, epochs=30): 
        print("Started training: {}".format(str(datetime.datetime.now())))
        for _ in range(0, int(epochs)):
            self.clf.fit(df_attributes, df_target)
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

        predicted = self.clf.predict([lst])
        
        return predicted

