import numpy as np
from util import dose_to_label
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

class LinUCB:        
    
    def __init__(self, num_features, num_labels, alpha=1.0):
        self.num_features = num_features
        self.num_labels = num_labels
        self.arms = list(range(1, self.num_labels+1))
        self.alpha = alpha
        self.reset()
            
    def reset(self):
        self.A = {i:np.eye(self.num_features) for i in self.arms}
        self.b = {i:np.zeros(self.num_features) for i in self.arms}
            
    def predict(self, fea, lab=None, t=None):
        fea = fea / np.linalg.norm(fea)
        p = {}
        for arm in self.arms:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])
            p[arm] = theta.dot(fea) + self.alpha * np.sqrt(fea.dot(A_inv).dot(fea))
        # pred = max(p, key=p.get)
        pred = np.random.choice([key for key, value in p.items() if value == max(p.values())])
        return pred
        
    def update(self, fea, pred, reward):
        fea = fea / np.linalg.norm(fea)
        self.A[pred] += np.outer(fea, fea)
        self.b[pred] += reward * fea


class FixedDose:        
    
    def __init__(self, dose=.0):
        self.dose = dose
            
    def reset(self):
        pass
            
    def predict(self, fea=None, lab=None, t=None):
        return dose_to_label(self.dose)
    
    def update(self, fea, pred, reward):
        pass
    
        
class ClinicalDose:        
    
    def __init__(self, cols):
        self.cols = cols
            
    def reset(self):
        pass
            
    def predict(self, clinical_dose_fea, lab=None, t=None):
        clinical_dose_fea = dict(zip(self.cols,clinical_dose_fea))
        self.dose = 4.0376 - \
            0.2546 * clinical_dose_fea['Age in decades'] + \
            0.0118 * clinical_dose_fea['Height in cm'] + \
            0.0134 * clinical_dose_fea['Weight in kg'] - \
            0.6752 * clinical_dose_fea['Asian Race'] + \
            0.4060 * clinical_dose_fea['Black or African American'] + \
            0.0443 * clinical_dose_fea['Missing or Mixed Race'] + \
            1.2799 * clinical_dose_fea['Enzyme Inducer Status'] - \
            0.5695 * clinical_dose_fea['Amiodarone Status']
        self.dose = self.dose ** 2
        return dose_to_label(self.dose)
    
    def update(self, fea, pred, reward):
        pass
             
             
class LassoUCB:
    
    def __init__(self, num_features, num_labels, num_samples, q=1, h=5, lambda1=0.05, lambda2_0=0.05):
        self.num_features = num_features
        self.num_labels = num_labels
        self.num_samples = num_samples
        self.arms = list(range(1, self.num_labels+1))
        self.q = q
        self.h = h
        self.lambda1 = lambda1
        self.lambda2_0 = lambda2_0
    
    def reset(self):
        self.obs_fea = []
        self.obs_lab = []
        self.T = self.S = {i:[] for i in self.arms}
        self.b_T = self.b_S = {i:np.zeros(self.num_features) for i in self.arms}
        self.intercept_T = self.intercept_S = {i:0 for i in self.arms}
        self.T_all = {}
        self.lambda2 = self.lambda2_0
        for i in self.arms:
            j_s = [self.q * (i - 1) + k for k in range(1, self.q + 1)]
            self.T_all[i] = [(2 ** n - 1) * self.num_labels * self.q + j for n in range(self.num_samples) for j in j_s 
                         if (2 ** n - 1) * self.num_labels * self.q + j <= self.num_samples]
    
    def predict(self, fea, lab, t):
        simplefilter("ignore", category=ConvergenceWarning)
        self.obs_fea.append(fea)
        self.obs_lab.append(lab)
        obs_fea = np.array(self.obs_fea)
        obs_lab = np.array(self.obs_lab)
        if t > 0:
            for arm in self.arms:
                if self.T[arm]:
                    X = obs_fea[self.T[arm]]
                    Y = obs_lab[self.T[arm]] 
                    lasso_T = Lasso(alpha=self.lambda1, max_iter=3000)
                    lasso_T.fit(X, Y)
                    self.b_T[arm] = lasso_T.coef_
                    self.intercept_T[arm] = lasso_T.intercept_
                    lasso_S = Lasso(alpha=self.lambda2, max_iter=3000)
                    lasso_S.fit(X, Y)
                    self.b_S[arm] = lasso_S.coef_  
                    self.intercept_S[arm] = lasso_S.intercept_
                    del lasso_T, lasso_S
        forced = False
        for arm in self.arms:
            if t + 1 in self.T_all[arm]:
                self.T[arm].append(t)
                self.S[arm].append(t)
                pred = t
                forced = True
        if not forced:
            max_forced = max(fea.dot(self.b_T[arm]) + self.intercept_T[i] for i in self.arms)
            kappa = [arm for arm in self.arms if fea.dot(self.b_T[arm] + self.intercept_T[arm]) >= max_forced - self.h / 2]
            p = {arm:fea.dot(self.b_S[arm] + self.intercept_S[arm]) for arm in kappa}
            # pred = max(p, key=p.get)
            pred = np.random.choice([key for key, value in p.items() if value == max(p.values())])
        self.S[pred].append(t)
        self.lambda2 = self.lambda2_0 * np.sqrt((np.log(t + 1) + np.log(self.num_features)) / (t + 1))
        return pred     

    def update(self, fea, pred, reward):
        pass
        
        
        
        