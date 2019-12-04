#Helper class to handle game data from a series of games
class Training_Data_Collector:
    def __init__(self, score_threshold, target_good_games):
        self.attributes = []
        self.targets = []
        self.scores = []
        self.accepted_scores = []
        self.score_threshold = score_threshold
        self.target_good_games = target_good_games
    
    def add_game(self, game_memory, score):
        #game_memory is a list of lists of size 2. [][0] is observation [][1] is action 
        
        if score >= self.score_threshold: #If the overall game was good, then train on its moves
            self.accepted_scores.append(score)
            #output = []
            for data in game_memory:
                self.attributes.append(data[0])
                self.targets.append(data[1])
        self.scores.append(score)
    
    def get_training_data(self):
        return self.attributes, self.targets
    
    def get_score(self):
        total_sum = 0
        for x in self.scores:
            total_sum += x
        accepted_sum = 0
        for x in self.accepted_scores:
            accepted_sum += x
        return accepted_sum // len(self.accepted_scores) , total_sum / len(self.scores)
        
    def get_stats(self):
        stats = {}
        
        stats["total games"] = len(self.scores)
        stats["total good games"] = len (self.accepted_scores)
        
        total_sum = 0
        for x in self.scores:
            total_sum += x
        accepted_sum = 0
        for x in self.accepted_scores:
            accepted_sum += x

        stats["avg good game score"] = accepted_sum / stats["total good games"]
        stats["avg game score"] = total_sum / stats["total games"]
        
        return stats

    def at_target(self):
        if len(self.accepted_scores) < self.target_good_games or len(self.scores) < 100:
            return False
        return True