#Notebook 4

from sklearn.linear_model import LogisticRegression
import datetime

class LR:
    def __init__(self):
        self.clf = LogisticRegression(solver='lbfgs', multi_class="multinomial", max_iter=1000, random_state=1)

    def train(self, df_attributes, df_target, epochs=10):
        scores = []
        print("Starting training...")
        for i in range(1, epochs + 1):
            print("Epoch:" + str(i) + "/" + str(epochs) + " -- " + str(datetime.datetime.now()))
            self.clf.fit(df_attributes, df_target)
            score = self.clf.score(df_attributes, df_target)
            scores.append(score)
        print("Done training.")
        return scores
    
    def get_next_move(self, current_state):
        predicted = self.clf.predict(current_state)
        return predicted

