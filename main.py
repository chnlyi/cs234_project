import argparse
import numpy as np
import pickle
from tqdm import tqdm
from process_data import process_data, process_data_clinical_dose, process_data_pharmacogenetic_dose
from policy import LinUCB, FixedDose, ClinicalDose, LassoUCB, RobustLinExp3, LinTS, PharmacogeneticDose, Supervised
from bandit import ContextualBandit
from plot import plot_ci

def run(bandit, policy, T=1000, num_trials=3, seed=2023, verbose=True):
    
    disable_tqdm = not verbose
    
    cum_regrets = []
    incorrectness = []
    np.random.seed(seed)
    
    for i, trial_seed in enumerate(np.random.randint(1, 100, num_trials)):
        
        bandit.reset(seed=trial_seed)
        policy.reset()
        
        for t in tqdm(range(T), desc=f'Seed: {trial_seed}', disable=disable_tqdm):
            
            fea, lab = bandit.present()
            arm = policy.predict(fea, lab, t)
            reward = bandit.pull(arm)
            policy.update(fea, arm, reward)
            
        if verbose:
            print(f'Trial #{i+1} for ({policy.__class__.__name__}) with Random Seed: {trial_seed}')
            print(f'Total Rewards ({policy.__class__.__name__}): {sum(bandit.rewards)}; Incorrectness: {bandit.incorrectness[-1]}')
        
        cum_regrets.append(np.cumsum(bandit.regrets))
        incorrectness.append(bandit.incorrectness)
    
    return cum_regrets, incorrectness

def main(features, labels, 
         clinical_features, clinical_cols, clinical_labels, 
         pharmacogenetic_features, pharmacogenetic_cols, pharmacogenetic_labels, 
         reward_mode='constant', reward_unit=3, T=1000, num_trials=3, seed=2023, verbose=True):
    
    if reward_mode == 'constant':
        reward_unit = 1
    
    np.random.seed(seed)
    
    linucb_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    fixed_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    clinical_cb = ContextualBandit(features=clinical_features, labels=clinical_labels, reward_mode=reward_mode, reward_unit=reward_unit)
    pharma_cb = ContextualBandit(features=pharmacogenetic_features, labels=pharmacogenetic_labels, reward_mode=reward_mode, reward_unit=reward_unit)
    lints_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    lasso_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    # robust_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
    sup_cb = ContextualBandit(features=features, labels=labels, reward_mode=reward_mode, reward_unit=reward_unit)
        
    bandits = [
        linucb_cb,
        fixed_cb,
        clinical_cb,
        pharma_cb,
        lints_cb,
        lasso_cb,
        # robust_cb,
        sup_cb,
    ]
    
    policies = [
        LinUCB(num_features=linucb_cb.num_features, num_labels=linucb_cb.k),
        FixedDose(dose=35),
        ClinicalDose(cols=clinical_cols),
        PharmacogeneticDose(cols=pharmacogenetic_cols),
        LinTS(num_features=lints_cb.num_features, num_labels=lints_cb.k),
        LassoUCB(num_features=lasso_cb.num_features, num_labels=lasso_cb.k, num_samples=T, q=1, h=5, lambda1=0.05, lambda2_0=0.05),
        # RobustLinExp3(all_features=robust_cb.features, num_labels=robust_cb.k, eta=0.01, gamma=0.1),
        Supervised(num_features=sup_cb.num_features, num_labels=sup_cb.k, lr=0.01),
    ]
    
    cum_regrets = {}
    incorrectness = {}
    for bandit, policy in zip(bandits, policies):
        cum_reg, incor = run(bandit, policy, T=T, num_trials=num_trials, seed=seed, verbose=verbose)
        cum_reg = [i / reward_unit for i in cum_reg]
        cum_regrets[policy.__class__.__name__] = np.array(cum_reg)
        incorrectness[policy.__class__.__name__] = np.array(incor)

    plot_ci(cum_regrets, title='Regrets by Timestep With 95% Confidence Bounds', save_path='figures/regret.png')
    plot_ci(incorrectness, title='Incorrectness by Timestep With 95% Confidence Bounds', ylim_low=0.25, ylim_high=0.6, save_path='figures/incorrectness.png')
    
    return cum_regrets, incorrectness

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="CS234 Final Project Linear Bandit")
    parser.add_argument("-p", "--path", type=str, default='data/warfarin.csv')
    parser.add_argument("-t", "--trials", type=int, default=3)
    parser.add_argument("-s", "--seed", type=int, default=2023)
    args = parser.parse_args()
    
    features, labels = process_data(args.path)
    clinical_features, clinical_cols, clinical_labels = process_data_clinical_dose(args.path)
    pharmacogenetic_features, pharmacogenetic_cols, pharmacogenetic_labels = process_data_pharmacogenetic_dose(args.path)
    
    cum_regrets, incorrectness = main(features, labels, 
         clinical_features, clinical_cols, clinical_labels,
         pharmacogenetic_features, pharmacogenetic_cols, pharmacogenetic_labels, 
         reward_mode='linear', reward_unit=3, 
         T=features.shape[0], num_trials=args.trials, seed=args.seed)
    
    save_path = 'save/results.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump([cum_regrets, incorrectness], file)