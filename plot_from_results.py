import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

###############################################################
def plot_final_incor_reward_risk(df, save_path):

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df, kind="bar",
        x="Reward Unit", 
        y="Final Incorrectness", 
        hue="Reward Mode", 
        col="Model",
        errorbar="sd", 
        palette="dark", 
        alpha=.6, 
        height=4, 
        aspect=4/4
    )
    g.despine(left=True)
    g.set_axis_labels("Reward Unit", "Incorrectness")
    g.legend.set_title("Reward Mode")  
    plt.savefig(save_path)
    plt.close()

def plot_incor_time_reward_risk(df, save_path):
    
    plot_dts = {}
    for model in df['Model'].unique():
        plot_dts[model] = {}
        for mode in df['Reward Mode'].unique():
            for unit in df['Reward Unit'].unique():
                dt = df.loc[(df['Model']==model)&(df['Reward Mode']==mode)&(df['Reward Unit']==unit)]['Incorrectness']
                plot_dts[model][mode+' unit '+str(unit)] = np.array([i for i in dt])

      
    fig, ax = plt.subplots(ncols=4, figsize=(20, 4.5), sharey=True)    
    for i, (algo_name, dicttt) in enumerate(plot_dts.items()):
        for model, dt in dicttt.items():
            T = dt.shape[1]
            means = np.apply_along_axis(np.mean, 0, dt)
            if model.startswith('exp'):
                linestyle = (0, (5, 5))
                linewidth = 1
            else:
                linestyle = None
                linewidth = 1
            ax[i].plot(range(T), means, label=model, linestyle=linestyle, linewidth=linewidth)
        ax[i].set_title(f'Model = {algo_name}', fontsize=16)
    fig.tight_layout() 
    fig.text(0.5, -0.04, 'Timestep', ha='center', fontsize=16)
    fig.text(-0.02, 0.5, 'Incorrectness', va='center', rotation='vertical', fontsize=16) 
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, .9), prop={'size':10})
    plt.ylim([0.3, 0.5])
    plt.savefig(save_path, bbox_inches = 'tight')
    plt.close()


path = 'plot_df.pkl'
with open(path, 'rb') as file:
    df = pickle.load(file)    
df['Final Incorrectness'] = df['Incorrectness'].apply(lambda x : x[-1])

plot_final_incor_reward_risk(df, save_path='figures/final_incor_rr.png')
plot_incor_time_reward_risk(df, save_path='figures/incor_time_rr.png')

################################################################################

def plot_final_bar(plt_df, y, save_path='figures/'):
    
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=plt_df,
        kind="bar",
        y='Model', 
        x=y, 
        errorbar="sd", 
        palette="dark",
        orient='h',
        alpha=.6, 
        height=4, 
        aspect=6.5/4
    )
    g.despine(left=True)
    g.set_axis_labels("Final Incorrectness", "Model")
    g.set_yticklabels(size=10)
    for container in g.ax.containers:
        g.ax.bar_label(container, fmt='%.3f', padding=6, fontsize=8) 
    plt.savefig(save_path)
    plt.close()


path = 'save/results.pkl'
with open(path, 'rb') as file:
    cum_regrets, incorrectness = pickle.load(file)
    
plt_df = pd.DataFrame(columns=['Model', 'Final Incorrectness', 'Final Regret'])
for model in cum_regrets:
    for incors, regs in zip(incorrectness[model], cum_regrets[model]):
        plt_df.loc[len(plt_df)] = [model, incors[-1], regs[-1]]

plot_final_bar(plt_df, y='Final Incorrectness', save_path='figures/final_incor_bar.png')
plot_final_bar(plt_df, y='Final Regret', save_path='figures/regret_bar.png')