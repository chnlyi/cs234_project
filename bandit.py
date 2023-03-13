import numpy as np
import pandas as pd

class ContextualBandit:
    
    def __init__(self, features, labels, linear_reward=True, reward_unit=1, seed=2023): 
                
        self.features = self.type_check(features)    
        self.labels = self.type_check(labels)
        self.seed = seed
        self.num_samples = features.shape[0]
        self.num_features = features.shape[1]        
        self.k = len(np.unique(self.labels))
        self.linear_reward = linear_reward # determine if the reward is linear or exponential w.r.t mistake
        self.reward_unit = reward_unit
        self.reset()
       
    @staticmethod
    def type_check(x):
        if isinstance(x, np.ndarray):
            return x
        else:
            if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
                return x.values
            else:
                raise TypeError("Please use Pandas DataFrame or Numpy Array!")        
        
    def reset(self, seed=None):
        self.features_copy = self.features.copy()
        self.labels_copy = self.labels.copy()
        self.sample_row = 0
        self.regrets = []
        self.rewards = []
        self.correctness = []
        self.mistakes = []
        self.pulls = []
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        np.random.shuffle(self.features_copy)
        np.random.seed(seed)
        np.random.shuffle(self.labels_copy)
        
    def present(self):
        self.current_feature = self.features_copy[self.sample_row, :]
        self.current_label = self.labels_copy[self.sample_row]
        self.sample_row += 1
        return self.current_feature, self.current_label
    
    def pull(self, arm):
        self.pulls.append(arm)
        mistake = abs(self.current_label - arm)
        if mistake == 0:
            reward = 0
        else:
            if self.linear_reward:
                reward = - self.reward_unit * mistake
            else:
                reward = - self.reward_unit ** (mistake - 1)
        regret = - reward
        self.rewards.append(reward)
        self.regrets.append(regret)
        num_correct = sum(i==0 for i in self.rewards)
        self.correctness.append(num_correct / len(self.rewards))
        self.mistakes.append(mistake)
        return reward