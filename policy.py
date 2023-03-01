import numpy as np
from util import dose_to_label

class LinUCB:        
    
    def __init__(self, num_features, num_labels, alpha=.0):
        self.num_features = num_features
        self.num_labels = num_labels
        self.arms = list(range(1, self.num_labels+1))
        self.alpha = alpha
        self.reset()
            
    def reset(self):
        self.A = {}
        self.b = {}
        for arm in self.arms:
            self.A[arm], self.b[arm] = np.eye(self.num_features), np.zeros(self.num_features)
            
    def predict(self, fea):
        fea = fea / np.linalg.norm(fea)
        p = {}
        for arm in self.arms:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])
            p[arm] = theta.dot(fea) + self.alpha * np.sqrt(fea.dot(A_inv).dot(fea))
        pred = max(p, key=p.get)
        return pred, self.b
        
    def update(self, fea, pred, reward):
        fea = fea / np.linalg.norm(fea)
        self.A[pred] += np.outer(fea, fea)
        self.b[pred] += reward * fea
        
        
class FixedDose:        
    
    def __init__(self, dose=.0):
        self.dose = dose
            
    def reset(self):
        pass
            
    def predict(self):
        return dose_to_label(self.dose)
    
        
class ClinicalDose:        
    
    def __init__(self, dose=None):
        self.dose = dose
            
    def reset(self):
        pass
            
    def predict(self, clinical_dose_fea):
        self.dose = 4.0376 - \
            0.2546 * clinical_dose_fea['Age in decades'].values[0] + \
            0.0118 * clinical_dose_fea['Height in cm'].values[0] + \
            0.0134 * clinical_dose_fea['Weight in kg'].values[0] - \
            0.6752 * clinical_dose_fea['Asian Race'].values[0] + \
            0.4060 * clinical_dose_fea['Black or African American'].values[0] + \
            0.0443 * clinical_dose_fea['Missing or Mixed Race'].values[0] + \
            1.2799 * clinical_dose_fea['Enzyme Inducer Status'].values[0] - \
            0.5695 * clinical_dose_fea['Amiodarone Status'].values[0]
        self.dose = self.dose ** 2
        # import pdb; pdb.set_trace()
        return dose_to_label(self.dose)
 