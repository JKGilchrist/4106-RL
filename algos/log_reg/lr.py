#Ref Notebook 4

from sklearn.linear_model import LogisticRegression
import datetime


class logistic_regression:
    def __init__(self, model=False):
        self.clf = LogisticRegression(solver='lbfgs', multi_class="multinomial", max_iter=1000, random_state=1)

    def train(self, df_attributes, df_target, epochs=50):
        print("Started training: {}".format(str(datetime.datetime.now())))
        for _ in range(1, epochs + 1):
            self.clf.fit(df_attributes, df_target)
        print("Finished training: {}".format(str(datetime.datetime.now())))
    
    def predict(self, prev_state):
        lst = []
        for x in prev_state:
            lst.append(x)
        predicted = self.clf.predict([lst])
        return predicted

