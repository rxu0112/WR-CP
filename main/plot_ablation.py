import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_list = ['airfoil', 'pemsd4', 'pemsd8', 'seattle', 'states', 'japan']
WRCP_COVERAGE = []
WRCP_SIZE = []
IWCP_COVERAGE = []
IWCP_SIZE = []

versions = range(1, 11)

for idx, dataset in enumerate(dataset_list):

    WRCP_COVERAGE.append({})
    WRCP_COVERAGE[idx]['dataset'] = dataset
    WRCP_SIZE.append({})
    WRCP_SIZE[idx]['dataset'] = dataset

    IWCP_COVERAGE.append({})
    IWCP_COVERAGE[idx]['dataset'] = dataset
    IWCP_SIZE.append({})
    IWCP_SIZE[idx]['dataset'] = dataset

    wrcp_overall_gap = []  # (#weight,1)
    wrcp_overall_size = []  # (#weight,1)
    iwcp_overall_gap = []  # (#weight,1)
    iwcp_overall_size = []  # (#weight,1)

    for v in versions:
        file_path = os.path.join(project_path, "main/main_result", dataset, f"v{v}")
        RESULT = pickle.load(open(file_path, "rb"))

        wrcp_overall_gap_ver = []
        wrcp_overall_size_ver = []
        iwcp_overall_gap_ver = []
        iwcp_overall_size_ver = []
        
        for i in RESULT:
            if i['opt'] == 'wr':
                wrcp_overall_gap_ver.append(i['test_overall_gap'])
                wrcp_overall_size_ver.append(i['test_overall_size'])

        wrcp_overall_gap.append(wrcp_overall_gap_ver)
        wrcp_overall_size.append(wrcp_overall_size_ver)
        
        for i in RESULT:
            if i['opt'] == 'erm':
                if i['cp'] == 'iw-scp':
                    iwcp_overall_gap_ver.append(i['test_overall_gap'])
                    iwcp_overall_size_ver.append(i['test_overall_size'])

        iwcp_overall_gap.append(iwcp_overall_gap_ver)
        iwcp_overall_size.append(iwcp_overall_size_ver)

    wrcp_weights = []
    for i in RESULT:
        if i['opt'] == 'wr':
            wrcp_weights.append(i['weight'])
    print(np.sort(wrcp_weights))
    WRCP_COVERAGE[idx]['mean'] = np.mean(wrcp_overall_gap, axis=0)
    WRCP_COVERAGE[idx]['std'] = np.std(wrcp_overall_gap, axis=0)

    WRCP_SIZE[idx]['mean'] = np.mean(wrcp_overall_size, axis=0)
    WRCP_SIZE[idx]['std'] = np.std(wrcp_overall_size, axis=0)

    IWCP_COVERAGE[idx]['mean'] = np.mean(iwcp_overall_gap, axis=0)
    IWCP_COVERAGE[idx]['std'] = np.std(iwcp_overall_gap, axis=0)

    IWCP_SIZE[idx]['mean'] = np.mean(iwcp_overall_size, axis=0)
    IWCP_SIZE[idx]['std'] = np.std(iwcp_overall_size, axis=0)

fig, axs = plt.subplots(1, 6, figsize=(13.8, 2.7))
fig.subplots_adjust(left=0.05, bottom=0.338, right=0.998, top=0.876, hspace=0.083, wspace=0.274)
titles = ['Airfoil', 'PeMSD4', 'PeMSD8', 'Seattle', 'US', 'Japan']
labels = ['IW-CP', 'WR-CP']
colors = ['orange', 'blue']
for i in range(6):
    axs[i].errorbar(WRCP_SIZE[i]['mean'], WRCP_COVERAGE[i]['mean'],
                       xerr=WRCP_SIZE[i]['std'], yerr=WRCP_COVERAGE[i]['std'],fmt="o")
    axs[i].errorbar(IWCP_SIZE[i]['mean'], IWCP_COVERAGE[i]['mean'],
                       xerr=IWCP_SIZE[i]['std'], yerr=IWCP_COVERAGE[i]['std'],fmt="o")
    axs[i].title.set_text(titles[i])
    axs[i].title.set_fontsize(12)
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[i].set_xlabel('Pred. Set Size', fontsize=12)
    if i == 0:
        axs[i].set_ylabel('Coverage Gap', fontsize=12)
legend_patches = [
    mpatches.Patch(color='C1', label='IW-CP'),  # C0 is the default color for the first label
    mpatches.Patch(color='C0', label='WR-CP')   # C1 is the default color for the second label
]
fig.legend(handles=legend_patches, loc='lower center', ncol=2, prop={'size': 12})
plt.show()

