import argparse
import numpy as np
from tqdm import tqdm
from process_data import process_data, process_data_clinical_dose
from policy import LinUCB, FixedDose, ClinicalDose, LassoUCB
from bandit import ContextualBandit
from plot import plot_ci

def run (bandit, policy, T=1000, num_trials=3, seed=2023):
    
    cum_regrets = []
    correctness = []
    np.random.seed(seed)
    
    for i, trial_seed in enumerate(np.random.randint(1, 100, num_trials)):
        
        bandit.reset(seed=trial_seed)
        policy.reset()
        
        for t in tqdm(range(T)):
            
            fea, lab = bandit.present()
            arm = policy.predict(fea, lab, t)
            reward = bandit.pull(arm)            
            policy.update(fea, arm, reward)
            
        print(f'Trial #{i+1} for ({policy.__class__.__name__}) with Random Seed: {trial_seed}')
        print(f'Total Rewards ({policy.__class__.__name__}): {sum(bandit.rewards)}; Correctness: {bandit.correctness[-1]}')
        
        cum_regrets.append(np.cumsum(bandit.regrets))
        correctness.append(bandit.correctness)
    
    return cum_regrets, correctness

def main(features, labels, clinical_dose_df, clinical_dose_cols, clinical_dose_labels, 
         T=1000, num_trials=3, seed=2023):
    
    np.random.seed(seed)
    
    linucb_cb = ContextualBandit(features=features, labels=labels, reward_unit=10)
    fixed_cb = ContextualBandit(features=features, labels=labels, reward_unit=10)
    clinical_cb = ContextualBandit(features=clinical_dose_df, labels=clinical_dose_labels, reward_unit=10)
    lasso_cb = ContextualBandit(features=features, labels=labels, reward_unit=10)
        
    bandits = [
        linucb_cb,
        fixed_cb,
        clinical_cb,
        lasso_cb,
    ]
    
    policies = [
        LinUCB(num_features=linucb_cb.num_features, num_labels=linucb_cb.k, alpha=1.0),
        FixedDose(dose=35),
        ClinicalDose(cols=clinical_dose_cols),
        LassoUCB(num_features=lasso_cb.num_features, num_labels=lasso_cb.k, num_samples=T, q=1, h=5, lambda1=0.05, lambda2_0=0.05),
    ]
    
    cum_regrets = {}
    correctness = {}
    for bandit, policy in zip(bandits, policies):
        cum_reg, cor = run(bandit, policy, T=T, num_trials=num_trials, seed=seed)
        cum_regrets[policy.__class__.__name__] = np.array(cum_reg)
        correctness[policy.__class__.__name__] = np.array(cor)
        
    plot_ci(cum_regrets, save_path='figures/regret.png')
    plot_ci(correctness, save_path='figures/correctness.png')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="CS234 Final Project Linear Bandit")
    parser.add_argument("-p", "--path", type=str, default='data/warfarin.csv')
    parser.add_argument("-t", "--trials", type=int, default=3)
    parser.add_argument("-s", "--seed", type=int, default=2023)
    args = parser.parse_args()
    
    features, labels = process_data(args.path)
    clinical_dose_features, clinical_dose_cols, clinical_dose_labels = process_data_clinical_dose(args.path)
    main(features, labels, clinical_dose_features, clinical_dose_cols, clinical_dose_labels, 
         T=features.shape[0], num_trials=args.trials, seed=args.seed)