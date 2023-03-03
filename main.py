import argparse
import numpy as np
from tqdm import tqdm
from process_data import process_data, process_data_clinical_dose
from policy import LinUCB, FixedDose, ClinicalDose
from bandit import ContextualBandit
from plot import plot_ci

def main(features, labels, dosage, clinical_dose_df, T=1000, num_trials=3, alpha=.0, seed=2023):
    
    np.random.seed(seed)
    
    linucb_cb = ContextualBandit(features=features, labels=labels, dosage=dosage)
    fixed_cb = ContextualBandit(features=features, labels=labels, dosage=dosage)
    clinical_dose_cb = ContextualBandit(features=features, labels=labels, dosage=dosage, clinical_dose_df=clinical_dose_df)
    
    linucb = LinUCB(num_features=linucb_cb.num_features, num_labels=linucb_cb.k, alpha=alpha)
    fixed = FixedDose(dose=35)
    clinical_dose = ClinicalDose()
    
    cum_regrets = []
    linucb_correctness = []
    fixed_correctness = []
    clinical_dose_correctness = []
    
    for i, trial_seed in enumerate(np.random.randint(1, 100, num_trials)):
        
        linucb_cb.reset(seed=trial_seed)
        fixed_cb.reset(seed=trial_seed)
        clinical_dose_cb.reset(seed=trial_seed)
        
        linucb.reset()
        
        for t in tqdm(range(T)):
            
            fea, _1 = linucb_cb.present_features()
            arm, b = linucb.predict(fea)
            reward = linucb_cb.pull(arm, b=b)            
            linucb.update(fea, arm, reward)
            
            fixed_fea, _2 = fixed_cb.present_features()
            fixed_arm = fixed.predict()
            fixed_reward = fixed_cb.pull(fixed_arm)
            
            _3, clinical_dose_fea = clinical_dose_cb.present_features()
            clinical_dose_arm = clinical_dose.predict(clinical_dose_fea)
            clinical_dose_reward = clinical_dose_cb.pull(clinical_dose_arm)
            
        print(f'Trial #{i+1} Random Seed: {trial_seed}')
        print(f'Total Rewards (FixedDose): {sum(fixed_cb.rewards)}; Correctness (FixedDose): {fixed_cb.correctness[-1]}')
        print(f'Total Rewards (ClinicalDose): {sum(clinical_dose_cb.rewards)} Correctness (ClinicalDose): {clinical_dose_cb.correctness[-1]}')
        print(f'Total Rewards (LinUCB): {sum(linucb_cb.rewards)} Correctness (LinUCB): {linucb_cb.correctness[-1]}')
        
        cum_regrets.append(linucb_cb.cum_regrets)
        linucb_correctness.append(linucb_cb.correctness)
        fixed_correctness.append(fixed_cb.correctness)
        clinical_dose_correctness.append(clinical_dose_cb.correctness)
    
    cum_regrets = {'LinUCB': np.array(cum_regrets)}
    correctness = {'LinUCB': np.array(linucb_correctness), 'FixedDose': np.array(fixed_correctness), 'ClinicalDose': np.array(clinical_dose_correctness)}
    
    plot_ci(cum_regrets, save_path='figures/regret.png')
    plot_ci(correctness, save_path='figures/correctness.png')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="CS234 Final Project Linear Bandit")
    parser.add_argument("-p", "--path", type=str, default='data/warfarin.csv')
    parser.add_argument("-t", "--trials", type=int, default=3)
    parser.add_argument("-a", "--alpha", type=float, default=1.0)
    parser.add_argument("-s", "--seed", type=int, default=2023)
    args = parser.parse_args()
    
    ids, features, labels, dosage, feature_names = process_data(args.path)
    clinical_dose_df = process_data_clinical_dose(args.path)
    main(features, labels, dosage, clinical_dose_df, T=features.shape[0], num_trials=args.trials, alpha=args.alpha, seed=args.seed)