import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_list = ['airfoil', 'pemsd4', 'pemsd8', 'seattle', 'states', 'japan']
dataset_weight = [9, 11, 9, 10, 13, 13]

versions = range(1, 11)
DISTANCE = []
for idx, dataset in enumerate(dataset_list):
    DISTANCE.append({})
    DISTANCE[idx]['dataset'] = dataset

    iwcp_W = []
    scp_W = []
    wrcp_W = []

    for v in versions:
        file_path = os.path.join(project_path, "main/main_result", dataset, f"v{v}")
        RESULT = pickle.load(open(file_path, "rb"))

        for i in RESULT:
            if i['cp'] == 'scp':
                scp_W.append(i['test_W'] / i['test_res'])
            if i['opt'] == 'erm':
                if i['cp'] == 'iw-scp':
                    iwcp_W.append(i['test_W'] / i['test_res'])

        wrcp_weights = []
        for i in RESULT:
            if i['opt'] == 'wr':
                wrcp_weights.append(i['weight'])

        for i in RESULT:
            if i['opt'] == 'wr':
                if i['weight'] == dataset_weight[idx]:
                    wrcp_W.append(i['test_W'] / i['test_res'])

    print(np.sort(wrcp_weights))

    DISTANCE[idx]['mean'] = np.mean(np.stack([scp_W, iwcp_W, wrcp_W]), axis=1)
    DISTANCE[idx]['std'] = np.std(np.stack([scp_W, iwcp_W, wrcp_W]), axis=1)

colors = ['peachpuff', 'lightgreen', 'lightblue']
labels = ['Vanilla CP', 'IW-CP', 'WR-CP']
titles = ['Airfoil', 'PeMSD4', 'PeMSD8', 'Seattle', 'US', 'Japan']
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]

fig, axs = plt.subplots(1, 6, figsize=(13.8, 2.3))
fig.subplots_adjust(left=0.05, bottom=0.212, right=0.995, top=0.876, hspace=0.083, wspace=0.28)
for i in range(6):
    axs[i].bar([1, 2, 3], DISTANCE[i]['mean'], color=colors)
    plotline, caplines, barlinecols = axs[i].errorbar([1, 2, 3], DISTANCE[i]['mean'],
                                                      yerr=DISTANCE[i]['std'],
                                                      lolims=True, capsize=0, ls='None', color='k')
    caplines[0].set_marker('_')
    caplines[0].set_markersize(5)
    axs[i].title.set_text(titles[i])
    axs[i].title.set_fontsize(12)
    axs[i].set_xticks([])
    axs[i].set_xticklabels([])
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if i == 0:
        axs[i].set_ylabel('Normalized W', fontsize=12)

fig.legend(handles=legend_patches, loc='lower center', ncol=3, prop={'size': 12})
plt.show()
