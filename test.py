# from process_data import process_data, process_data_clinical_dose
# from bandit import ContextualBandit

# path = 'data/warfarin.csv'
# ids, features, labels, dosage, feature_names = process_data(path)
# clinical_dose_df = process_data_clinical_dose(path)
# cb = ContextualBandit(features=features, labels=labels, dosage=dosage)
# cb1 = ContextualBandit(features=features, labels=labels, dosage=dosage)
# cb2 = ContextualBandit(features=features, labels=labels, clinical_dose_df=clinical_dose_df, dosage=dosage)
# # print(features.shape)
# seed = 523609
# cb.reset(seed=seed)
# cb1.reset(seed=seed)
# cb2.reset(seed=seed)

# for _ in range(10):
#     fea, _ = cb.present_features()
#     fea1, _ = cb1.present_features()
#     fea2, _ = cb2.present_features()
#     # print(fea.shape, fea1.shape, fea2.shape)
#     print((fea==fea1).all(), (fea1==fea2).all(), _.values)
    


# import random
# import time

# from tqdm import tqdm

# def do_stuff():
#     time.sleep(0.01)
#     if random.randint(0, 10) < 3:
#         raise Exception()


# exception_count = 0
# with tqdm(
#     bar_format="Exceptions: {postfix} | Elapsed: {elapsed} | {rate_fmt}",
#     postfix=exception_count,
# ) as t:
#     for _ in range(1000):
#         try:
#             do_stuff()
#         except Exception:
#             exception_count += 1
#             t.postfix = exception_count
#             t.update()

# from sys import stdout
# from time import sleep
# for i in range(1,20):
#     stdout.write("\r%d" % i)
#     stdout.flush()
#     sleep(1)
# stdout.write("\n") 




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
    
x = np.random.normal(size=(3, 10))
# print(x)
print(np.apply_along_axis(np.mean, 0, x))
print(np.apply_along_axis(get_ci, 0, x, kind='low', confidence=0.9))
print(np.apply_along_axis(get_ci, 0, x, kind='high', confidence=0.9))