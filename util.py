import numpy as np
from scipy.stats import t

def get_ci(x, kind='low', confidence=0.95):
    m = x.mean()
    s = x.std()
    dof = len(x) - 1
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    if kind == 'low':
        ci = m-s*t_crit/np.sqrt(len(x))
    elif kind == 'high':
        ci = m+s*t_crit/np.sqrt(len(x))
    return ci

def dose_to_label(dose):
    if dose < 21:
        return 1
    elif 21 <= dose <= 49:
        return 2
    else:
        return 3