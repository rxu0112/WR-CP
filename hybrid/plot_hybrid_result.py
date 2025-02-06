import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import os

dataset_list = ['airfoil', 'pemsd4', 'pemsd8', 'seattle', 'states', 'japan']
versions = range(1, 11)
alpha_list = [0.1]

COVERAGE = []
SIZE = []
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for idx, dataset in enumerate(dataset_list):

    COVERAGE.append({})
    COVERAGE[idx]['dataset'] = dataset
    SIZE.append({})
    SIZE[idx]['dataset'] = dataset

    erm_wcscp_detailed_coverage = []
    erm_wcscp_detailed_size = []

    wr_iw_wc_detailed_coverage = []
    wr_iw_wc_detailed_size = []


    for v in versions:

        RESULT_1 = pickle.load(open(project_path + "/main/main_result/" + dataset + "/v" + str(v), "rb"))
        RESULT_2 = pickle.load(open(project_path + "/hybrid/hybrid_result/" + dataset + "/v" + str(v), "rb"))

        erm_wcscp_detailed_coverage_ver = []
        erm_wcscp_detailed_size_ver = []
        wr_iw_wc_detailed_coverage_ver = []
        wr_iw_wc_detailed_size_ver = []

        for i in RESULT_1:
            if i['opt'] == 'erm':
                if i['cp'] == 'wc-scp':
                    erm_wcscp_detailed_coverage_ver.append(i['test_detailed_coverage'])
                    erm_wcscp_detailed_size_ver.append(i['test_detailed_size'])

        for i in RESULT_2:
            if i['opt'] == 'wr':
                wr_iw_wc_detailed_coverage_ver.append(i['test_detailed_coverage'])  # average over envs
                wr_iw_wc_detailed_size_ver.append(i['test_detailed_size'])


        erm_wcscp_detailed_coverage.append(erm_wcscp_detailed_coverage_ver)
        erm_wcscp_detailed_size.append(erm_wcscp_detailed_size_ver)

        wr_iw_wc_detailed_coverage.append(wr_iw_wc_detailed_coverage_ver)
        wr_iw_wc_detailed_size.append(wr_iw_wc_detailed_size_ver)

    erm_wcscp_detailed_coverage = np.concatenate([erm_wcscp_detailed_coverage[i][0] for i in range(len(versions))])
    erm_wcscp_detailed_size = np.concatenate([erm_wcscp_detailed_size[i][0] for i in range(len(versions))])


    wr_iw_wc_weights = []
    for i in RESULT_2:
        if i['opt'] == 'wr':
            wr_iw_wc_weights.append(i['weight'])


    wr_iw_wc_detailed_coverage_by_weight = []
    wr_iw_wc_detailed_size_by_weight = []

    for w in range(len(wr_iw_wc_weights)):
        wr_iw_wc_detailed_coverage_by_weight.append(
            np.concatenate([wr_iw_wc_detailed_coverage[i][w] for i in range(len(versions))]))
        wr_iw_wc_detailed_size_by_weight.append(np.concatenate([wr_iw_wc_detailed_size[i][w] for i in range(len(versions))]))

    COVERAGE[idx]['wc'] = erm_wcscp_detailed_coverage
    COVERAGE[idx]['wr_iw_wc'] = wr_iw_wc_detailed_coverage_by_weight
    COVERAGE[idx]['wr_iw_wc weights'] = wr_iw_wc_weights

    SIZE[idx]['wc'] = erm_wcscp_detailed_size
    SIZE[idx]['wr_iw_wc'] = wr_iw_wc_detailed_size_by_weight
    SIZE[idx]['wr_iw_wc weights'] = wr_iw_wc_weights


# raw for alpha, column for dataset
selected_weight_list = [[9, 11, 9, 8, 13, 20]]



Size_reduction_wr_iw_wc = []
Coverage_wr_iw_wc = []
Coverage_desired = []

for a, alpha in enumerate(alpha_list):

    box_coverage = []
    box_size = []
    for idx, dataset in enumerate(dataset_list):
        selected_weight = selected_weight_list[a][idx]
        pos_wr_iw_wc = COVERAGE[idx]['wr_iw_wc weights'].index(selected_weight)
        box_coverage.append([COVERAGE[idx]['wc'][:, a], COVERAGE[idx]['wr_iw_wc'][pos_wr_iw_wc][:, a]])
        box_size.append([SIZE[idx]['wc'][:, a], SIZE[idx]['wr_iw_wc'][pos_wr_iw_wc][:, a]])
        Size_reduction_wr_iw_wc.append(np.mean(SIZE[idx]['wr_iw_wc'][pos_wr_iw_wc][:, a] / SIZE[idx]['wc'][:, a]))
        Coverage_wr_iw_wc.append(COVERAGE[idx]['wr_iw_wc'][pos_wr_iw_wc][:, a])
        Coverage_desired.append((1 - alpha) * np.ones(len(COVERAGE[idx]['wr_iw_wc'][pos_wr_iw_wc][:, a])))

    colors = ['lightpink', 'orange']
    labels = ['WC-CP', 'Hybrid WR-WC']
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    titles = ['Airfoil', 'PeMSD4', 'PeMSD8', 'Seattle', 'US', 'Japan']

    fig, axs = plt.subplots(2, 6, figsize=(13.8, 3.6))
    fig.subplots_adjust(left=0.05, bottom=0.14, right=0.995, top=0.905, hspace=0.083, wspace=0.31)

    for i in range(len(dataset_list)):
        box = axs[0, i].boxplot(box_coverage[i], showfliers=False, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Add a reference horizontal line
        axs[0, i].axhline(y=1 - alpha, color='r', linestyle='--')
        axs[0, i].title.set_text(titles[i])
        axs[0, i].title.set_fontsize(12)
        axs[0, i].set_xticks([])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_ylim(0.75, 1.05)


        if i == 0:
            axs[0, i].set_ylabel('Coverage',  fontsize=12)
    line_legend = mlines.Line2D([], [], color='red', linestyle='--', label=r'$1-\alpha$')

    for i in range(len(dataset_list)):
        avg = [np.mean(d) for d in box_size[i]]
        std = [np.std(d) for d in box_size[i]]
        axs[1, i].bar(range(1, len(box_size[i]) + 1), avg, color=colors)
        plotline, caplines, barlinecols = axs[1, i].errorbar(range(1, len(box_size[i]) + 1), avg,
                                                             yerr=std,
                                                             lolims=True, capsize=0, ls='None', color='k')
        caplines[0].set_marker('_')
        caplines[0].set_markersize(5)
        axs[1, i].set_xticks([])
        axs[1, i].set_xticklabels([])
        axs[1, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        if i == 0:
            axs[1, i].set_ylabel('Pred. Set Size',  fontsize=12)
    fig.legend(handles=legend_patches + [line_legend], loc='lower center', ncol=6, prop={'size': 12})

    plt.show()
