import numpy as np
from util import dose_to_label
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter

class LinUCB:        
    
    def __init__(self, num_features, num_labels, delta=.01):
        self.num_features = num_features
        self.num_labels = num_labels
        self.arms = list(range(1, self.num_labels+1))
        self.alpha = 1 + np.sqrt(np.log(2 / delta) / 2)
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
             
             
class PharmacogeneticDose:        
    
    def __init__(self, cols):
        self.cols = cols
            
    def reset(self):
        pass
            
    def predict(self, pharmacogenetic_dose_fea, lab=None, t=None):
        pharmacogenetic_dose_fea = dict(zip(self.cols, pharmacogenetic_dose_fea))
        self.dose = 5.6044 - \
            0.2614 * pharmacogenetic_dose_fea['Age in decades'] + \
            0.0087 * pharmacogenetic_dose_fea['Height in cm'] + \
            0.0128 * pharmacogenetic_dose_fea['Weight in kg'] - \
            0.8677 * pharmacogenetic_dose_fea['VKORC1 A/G'] - \
            1.6974 * pharmacogenetic_dose_fea['VKORC1 A/A'] - \
            0.4854 * pharmacogenetic_dose_fea['VKORC1 Unknown'] - \
            0.5211 * pharmacogenetic_dose_fea['CYP2C9 *1/*2'] - \
            0.9357 * pharmacogenetic_dose_fea['CYP2C9 *1/*3'] - \
            1.0616 * pharmacogenetic_dose_fea['CYP2C9 *2/*2'] - \
            1.9206 * pharmacogenetic_dose_fea['CYP2C9 *2/*2'] - \
            2.3312 * pharmacogenetic_dose_fea['CYP2C9 *3/*3'] - \
            0.2188 * pharmacogenetic_dose_fea['CYP2C9 Unknown'] - \
            0.1092 * pharmacogenetic_dose_fea['Asian Race'] - \
            0.2760 * pharmacogenetic_dose_fea['Black or African American'] - \
            0.1032 * pharmacogenetic_dose_fea['Missing or Mixed Race'] + \
            1.1816 * pharmacogenetic_dose_fea['Enzyme Inducer Status'] - \
            0.5503 * pharmacogenetic_dose_fea['Amiodarone Status']
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
                pred = arm
                forced = True
        if not forced:
            max_forced = max(fea.dot(self.b_T[arm]) + self.intercept_T[i] for i in self.arms)
            kappa = [arm for arm in self.arms if fea.dot(self.b_T[arm] + self.intercept_T[arm]) >= max_forced - self.h / 2]
            p = {arm:fea.dot(self.b_S[arm] + self.intercept_S[arm]) for arm in kappa}
            pred = np.random.choice([key for key, value in p.items() if value == max(p.values())])
            self.S[pred].append(t)
        self.lambda2 = self.lambda2_0 * np.sqrt((np.log(t + 1) + np.log(self.num_features)) / (t + 1))
        return pred     

    def update(self, fea, pred, reward):
        pass
        
        
class RobustLinExp3:
    
    def __init__(self, all_features, num_labels, eta=.1, gamma=.0):
        self.num_features = all_features.shape[1]
        self.num_labels = num_labels
        self.arms = list(range(1, self.num_labels+1))
        self.eta = eta
        self.gamma = gamma
        self.sigma = np.cov(all_features, rowvar=False)
        self.reset()
    
    def reset(self):
        self.theta = {i:np.zeros(self.num_features) for i in self.arms}
        self.cumloss = {i:0 for i in self.arms}
        self.pi = {i:0 for i in self.arms}
    
    def predict(self, fea, lab=None, t=None):
        w = {}
        total_w = 0
        for arm in self.arms:
            self.cumloss[arm] += fea.dot(self.theta[arm])
            w[arm] = np.exp(- self.eta * self.cumloss[arm])
            total_w += w[arm]
        low = 0
        high = 0
        r = np.random.uniform()
        pred = None
        for arm in sorted(self.arms):
            self.pi[arm] = (1 - self.gamma) * w[arm] / total_w + self.gamma / self.num_labels
            high += self.pi[arm]
            if low <= r < high:
                pred = arm
            low += self.pi[arm]   
        return pred      

    def update(self, fea, pred, reward=None):
        loss = fea.dot(self.theta[pred])
        cov_inv = np.linalg.inv(self.sigma)
        for arm in self.arms:
            if arm == pred:
                self.theta[arm] = fea.dot(cov_inv.T) * loss / self.pi[arm]
            else:
                self.theta[arm] = np.zeros(self.num_features)
                
                
class LinTS:
    
    def __init__(self, num_features, num_labels, nu=0.03):
        self.num_features = num_features
        self.num_labels = num_labels
        self.arms = list(range(1, self.num_labels+1))
        self.nu = nu
        self.reset()
    
    def reset(self):
        self.mu_hat = {i:np.zeros(self.num_features) for i in self.arms}
        self.B = {i:np.eye(self.num_features) for i in self.arms}
        self.f = {i:np.zeros(self.num_features) for i in self.arms}
    
    def predict(self, fea, lab=None, t=None):
        p = {}
        for arm in self.arms:
            B_inv = np.linalg.inv(self.B[arm])
            mu_t = np.random.default_rng().multivariate_normal(self.mu_hat[arm], self.nu ** 2 * B_inv, method='cholesky')
            # mu_t = np.random.multivariate_normal(self.mu_hat[arm], self.nu ** 2 * B_inv)
            p[arm] = fea.dot(mu_t)
        pred = np.random.choice([key for key, value in p.items() if value == max(p.values())])
        return pred  
        
    def update(self, fea, pred, reward):
        self.B[pred] += np.outer(fea, fea)
        self.f[pred] += reward * fea
        B_inv = np.linalg.inv(self.B[pred])
        self.mu_hat[pred] = self.f[pred].dot(B_inv.T)
