import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from process_data import process_data
from policy import LinUCB, LassoUCB, LinTS, Supervised
from bandit import ContextualBandit

def single_run(bandit, policy, T=1000, num_trials=3, seed=2023):
    incorrectness = []
    np.random.seed(seed)
    model_name = policy.__class__.__name__
    for i, trial_seed in enumerate(np.random.randint(1, 100, num_trials)):
        bandit.reset(seed=trial_seed)
        policy.reset()
        for t in tqdm(range(T), desc=f'Model {model_name}, Trial {i+1}, Seed: {trial_seed}'):
            fea, lab = bandit.present()
            arm = policy.predict(fea, lab, t)
            reward = bandit.pull(arm)                      
            policy.update(fea, arm, reward)
        incorrectness.append(bandit.incorrectness)
        print(f'Incorrect Arms Pulled: {bandit.incorrectness[-1]:.1%}')
    return incorrectness

def run(features, labels, units, T=1000, num_trials=3, seed=2023):
    np.random.seed(seed)
    plot_df = pd.DataFrame(columns=['Model', 'Reward Mode', 'Reward Unit', 'Incorrectness'])
    for mode in ['linear', 'exponential']:
        for reward_unit in units:
            linucb_cb = ContextualBandit(features=features, labels=labels, reward_mode=mode, reward_unit=reward_unit)
            lasso_cb = ContextualBandit(features=features, labels=labels, reward_mode=mode, reward_unit=reward_unit)
            lints_cb = ContextualBandit(features=features, labels=labels, reward_mode=mode, reward_unit=reward_unit)
            sup_cb = ContextualBandit(features=features, labels=labels, reward_mode=mode, reward_unit=reward_unit)
            linucb = LinUCB(num_features=linucb_cb.num_features, num_labels=linucb_cb.k)
            lasso = LassoUCB(num_features=linucb_cb.num_features, num_labels=linucb_cb.k, num_samples=T, q=1, h=5, lambda1=0.05, lambda2_0=0.05)
            lints = LinTS(num_features=linucb_cb.num_features, num_labels=linucb_cb.k)
            sup = Supervised(num_features=linucb_cb.num_features, num_labels=linucb_cb.k)
            bandits = [
                linucb_cb, 
                lasso_cb, 
                lints_cb, 
                sup_cb,
            ]
            policys = [
                linucb, 
                lasso, 
                lints, 
                sup,
            ]            
            for bandit, policy in zip(bandits, policys):
                model_name = policy.__class__.__name__
                print(f'Running Reward Mode {mode}, Reward Unit {reward_unit}....')
                incors = single_run(bandit, policy, T=T, num_trials=num_trials, seed=seed)
                for incor in incors:
                    plot_df.loc[len(plot_df)] = [model_name, mode, str(reward_unit), incor]
    return plot_df

path = 'data/warfarin.csv'
features, labels = process_data(path)
units = [1, 3, 6, 10, 20]
plot_df = run(features, labels, units, T=features.shape[0], num_trials=5, seed=2020)
with open('plot_df.pkl', 'wb') as file:
    pickle.dump(plot_df, file)