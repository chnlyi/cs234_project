import numpy as np
import pandas as pd

class ContextualBandit:
    
    def __init__(self, features, labels, dosage, clinical_dose_df=None, reward_unit=1, seed=2023):
        self.features = features
        self.features_idx = slice(0, self.features.shape[1])     
        self.labels = labels
        self.labels_idx = self.features.shape[1] + 1
        self.dosage = dosage
        self.dosage_idx = self.features.shape[1]
        self.clinical_dose_df = clinical_dose_df
        if self.clinical_dose_df is not None:
            self.clinical_dose_features_idx = slice(self.features.shape[1]+2, self.features.shape[1]+2+self.clinical_dose_df.shape[1])
        self.seed = seed
        self.num_features = features.shape[1]
        self.k = self.labels.nunique()
        self.reward_unit = reward_unit
        self.reset()
        
    def reset(self, seed=None):
        if self.clinical_dose_df is not None:
            stack = (self.features, self.dosage.values.reshape((-1,1)), self.labels.values.reshape((-1,1)), self.clinical_dose_df.values)
        else:
            stack = (self.features, self.dosage.values.reshape((-1,1)), self.labels.values.reshape((-1,1)))
        self.data = np.column_stack(stack)
        self.sample_row = 0
        self.regrets = []
        self.cum_regrets = []
        self.current_cum_regret = 0
        self.rewards = []
        self.correctness = []
        self.mistakes = []
        self.pulls = []
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        np.random.shuffle(self.data)
    
    def present_features(self):
        self.current_features = self.data[self.sample_row, self.features_idx]
        self.current_label = self.data[self.sample_row, self.labels_idx]
        self.current_dose = self.data[self.sample_row, self.dosage_idx]
        if self.clinical_dose_df is not None:
            arr = self.data[self.sample_row, self.clinical_dose_features_idx]
            self.current_clinical_dose_features = pd.DataFrame([arr], columns=self.clinical_dose_df.columns)
        else:
            self.current_clinical_dose_features = None
        self.sample_row += 1
        return self.current_features, self.current_clinical_dose_features
    
    def pull(self, arm, b=None):
        self.pulls.append(arm)
        if b is not None:
            regret = self.current_features.dot(b[self.current_label]) - self.current_features.dot(b[arm])
            self.regrets.append(regret)
            self.current_cum_regret += regret
            self.cum_regrets.append(self.current_cum_regret)
        if self.current_label == arm:
            reward = mistake = 0
        else:
            mistake = abs(self.current_label - arm)
            reward = - mistake * self.reward_unit
        self.rewards.append(reward)
        self.correctness.append(1 + sum(self.rewards) / len(self.rewards))
        self.mistakes.append(mistake)
        return reward