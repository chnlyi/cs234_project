import numpy as np
import matplotlib.pyplot as plt
from util import get_ci

def plot_ci(plot_data, confidence=0.95, save_path='figures/myplot.png'):
    
    for model, dt in plot_data.items():
        T = dt.shape[1]
        means = np.apply_along_axis(np.mean, 0, dt)
        low_ci = np.apply_along_axis(get_ci, 0, dt, kind='low', confidence=confidence)
        high_ci = np.apply_along_axis(get_ci, 0, dt, kind='high', confidence=confidence)
        
        plt.plot(range(T), means, label=model)
        plt.fill_between(range(T), low_ci, high_ci, alpha=.2)
    
    plt.legend()   
    plt.savefig(save_path)
    plt.close()