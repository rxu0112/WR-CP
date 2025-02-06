import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_list = ['airfoil', 'pemsd4', 'pemsd8', 'seattle', 'states', 'japan']
versions = range(1, 11)

W_std = []
TV_std = []
KL_std = []
E_std = []

W_avg = []
TV_avg = []
KL_avg = []
E_avg = []

for idx, dataset in enumerate(dataset_list):
    corr = {'dataset': dataset, 'wasserstein_corr': [],
            'total_variation_corr': [],
            'KL_divergence_corr': [],
            'expectation_difference_corr': []}
    for v in versions:
        file_path = os.path.join(project_path, "correlation", "correlation_result", dataset, f"v{v}")
        result = pickle.load(open(file_path, "rb"))[0]
        corr['wasserstein_corr'].append(spearmanr(result['test_overall_gap'], result['wasserstein_distance']).statistic)
        corr['total_variation_corr'].append(spearmanr(result['test_overall_gap'], result['total_variation_distance']).statistic)
        corr['KL_divergence_corr'].append(spearmanr(result['test_overall_gap'], result['KL_divergence']).statistic)
        corr['expectation_difference_corr'].append(spearmanr(result['test_overall_gap'], result['expectation_difference']).statistic)

    W_std.append(np.std(corr['wasserstein_corr']))
    TV_std.append(np.std(corr['total_variation_corr']))
    KL_std.append(np.std(corr['KL_divergence_corr']))
    E_std.append(np.std(corr['expectation_difference_corr']))

    W_avg.append(np.mean(corr['wasserstein_corr']))
    TV_avg.append(np.mean(corr['total_variation_corr']))
    KL_avg.append(np.mean(corr['KL_divergence_corr']))
    E_avg.append(np.mean(corr['expectation_difference_corr']))

dataset_list.append('Average')
W_std.append(np.std(W_std))
TV_std.append(np.std(TV_std))
KL_std.append(np.std(KL_std))
E_std.append(np.std(E_std))

W_avg.append(np.mean(W_avg))
TV_avg.append(np.mean(TV_avg))
KL_avg.append(np.mean(KL_avg))
E_avg.append(np.mean(E_avg))

df_std = pd.DataFrame()
df_std['dataset'] = dataset_list
df_std['Wasserstein distance'] = W_std
df_std['Total variation distance'] = TV_std
df_std['KL divergence'] = KL_std
df_std['Expectation difference'] = E_std

df_avg = pd.DataFrame()
df_avg['dataset'] = dataset_list
df_avg['Wasserstein distance'] = W_avg
df_avg['Total variation distance'] = TV_avg
df_avg['KL divergence'] = KL_avg
df_avg['Expectation difference'] = E_avg

std_path = os.path.join(project_path, "correlation", "correlation_result", "std.xlsx")
avg_path = os.path.join(project_path, "correlation", "correlation_result", "avg.xlsx")

df_std.to_excel(std_path, index=False, engine='openpyxl')
df_avg.to_excel(avg_path, index=False, engine='openpyxl')
