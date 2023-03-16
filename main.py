import argparse
import numpy as np
from tqdm import tqdm
from process_data import process_data, process_data_clinical_dose, process_data_pharmacogenetic_dose
from policy import LinUCB, FixedDose, ClinicalDose, LassoUCB, RobustLinExp3, LinTS, PharmacogeneticDose, Supervised
from bandit import ContextualBandit
from plot import plot_ci

def run(bandit, policy, T=1000, num_trials=3, seed=2023):
    
    cum_regrets = []
    correctness = []
    np.random.seed(seed)
    
    for i, trial_seed in enumerate(np.random.randint(1, 100, num_trials)):
        
        bandit.reset(seed=trial_seed)
        policy.reset()
        
        for t in tqdm(range(T), desc=f'Seed: {trial_seed}'):
            
            fea, lab = bandit.present()
            arm = policy.predict(fea, lab, t)
            reward = bandit.pull(arm)
            if policy.__class__.__name__ in ['RobustLinExp3']:
                policy.update(fea, arm, reward) 
            else:                           
                policy.update(fea, arm, reward)
            
        print(f'Trial #{i+1} for ({policy.__class__.__name__}) with Random Seed: {trial_seed}')
        print(f'Total Rewards ({policy.__class__.__name__}): {sum(bandit.rewards)}; Correctness: {bandit.correctness[-1]}')
        
        cum_regrets.append(np.cumsum(bandit.regrets))
        correctness.append(bandit.correctness)
    
    return cum_regrets, correctness

def main(features, labels, 
         clinical_features, clinical_cols, clinical_labels, 
         pharmacogenetic_features, pharmacogenetic_cols, pharmacogenetic_labels, 
         reward_mode='constant', reward_unit=3, T=1000, num_trials=3, seed=2023):
    
    if reward_mode == 'constant':
        reward_unit = 1
    
    np.random.seed(seed)
    
    # linucb_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    # fixed_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    # clinical_cb = ContextualBandit(features=clinical_features, labels=clinical_labels, reward_mode=reward_mode, reward_unit=reward_unit)
    # pharma_cb = ContextualBandit(features=pharmacogenetic_features, labels=pharmacogenetic_labels, reward_mode=reward_mode, reward_unit=reward_unit)
    # lints_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    # lasso_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    # robust_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    sup_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
        
    bandits = [
        # linucb_cb,
        # fixed_cb,
        # clinical_cb,
        # pharma_cb,
        # lints_cb,
        # lasso_cb,
        # robust_cb,
        sup_cb,
    ]
    
    policies = [
        # LinUCB(num_features=linucb_cb.num_features, num_labels=linucb_cb.k),
        # FixedDose(dose=35),
        # ClinicalDose(cols=clinical_cols),
        # PharmacogeneticDose(cols=pharmacogenetic_cols),
        # LinTS(num_features=lints_cb.num_features, num_labels=lints_cb.k),
        # LassoUCB(num_features=lasso_cb.num_features, num_labels=lasso_cb.k, num_samples=T, q=1, h=5, lambda1=0.05, lambda2_0=0.05),
        # RobustLinExp3(all_features=robust_cb.features, num_labels=robust_cb.k, eta=0.01, gamma=0.1),
        Supervised(num_features=sup_cb.num_features, num_labels=sup_cb.k, lr=0.01),
    ]
    
    cum_regrets = {}
    correctness = {}
    for bandit, policy in zip(bandits, policies):
        cum_reg, cor = run(bandit, policy, T=T, num_trials=num_trials, seed=seed)
        cum_reg = [i / reward_unit for i in cum_reg]
        cum_regrets[policy.__class__.__name__] = np.array(cum_reg)
        correctness[policy.__class__.__name__] = np.array(cor)
     
    plot_ci(cum_regrets, save_path='figures/regret.png')
    plot_ci(correctness, ylim_low=0.4, ylim_high=0.7, save_path='figures/correctness.png')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="CS234 Final Project Linear Bandit")
    parser.add_argument("-p", "--path", type=str, default='data/warfarin.csv')
    parser.add_argument("-t", "--trials", type=int, default=3)
    parser.add_argument("-s", "--seed", type=int, default=2023)
    args = parser.parse_args()
    
    features, labels = process_data(args.path)
    clinical_features, clinical_cols, clinical_labels = process_data_clinical_dose(args.path)
    pharmacogenetic_features, pharmacogenetic_cols, pharmacogenetic_labels = process_data_pharmacogenetic_dose(args.path)
    
    main(features, labels, 
         clinical_features, clinical_cols, clinical_labels,
         pharmacogenetic_features, pharmacogenetic_cols, pharmacogenetic_labels, 
         reward_mode='constant', reward_unit=1, 
         T=features.shape[0], num_trials=args.trials, seed=args.seed)