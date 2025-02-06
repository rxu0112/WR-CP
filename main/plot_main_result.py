import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_list = ['airfoil', 'pemsd4', 'pemsd8', 'seattle', 'states', 'japan']
versions = range(1, 11)
alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

COVERAGE = []
SIZE = []

for idx, dataset in enumerate(dataset_list):

    COVERAGE.append({})
    COVERAGE[idx]['dataset'] = dataset
    SIZE.append({})
    SIZE[idx]['dataset'] = dataset

    wrcp_overall_gap = []  # (#weight,1)
    wrcp_overall_size = []  # (#weight,1)
    wrcp_detailed_coverage = []  # (#weight,#alpha)
    wrcp_detailed_size = []  # (#weight,#alpha)

    erm_scp_detailed_coverage = []  # (#alpha, 1)
    erm_scp_detailed_size = []  # (#alpha, 1)
    erm_wcscp_detailed_coverage = []
    erm_wcscp_detailed_size = []
    erm_iwscp_detailed_coverage = []
    erm_iwscp_detailed_size = []
    cqr_detailed_coverage = []
    cqr_detailed_size = []

    for v in versions:

        file_path = os.path.join(project_path, "main/main_result", dataset, f"v{v}")
        RESULT = pickle.load(open(file_path, "rb"))

        wrcp_overall_gap_ver = []
        wrcp_overall_size_ver = []
        wrcp_detailed_coverage_ver = []
        wrcp_detailed_size_ver = []

        erm_scp_detailed_coverage_ver = []
        erm_scp_detailed_size_ver = []
        erm_wcscp_detailed_coverage_ver = []
        erm_wcscp_detailed_size_ver = []
        erm_iwscp_detailed_coverage_ver = []
        erm_iwscp_detailed_size_ver = []
        cqr_detailed_coverage_ver = []
        cqr_detailed_size_ver = []

        for i in RESULT:
            if i['opt'] == 'wr':
                wrcp_overall_gap_ver.append(i['test_overall_gap'])
                wrcp_overall_size_ver.append(i['test_overall_size'])
                wrcp_detailed_coverage_ver.append(i['test_detailed_coverage'])  # average over envs
                wrcp_detailed_size_ver.append(i['test_detailed_size'])

            if i['opt'] == 'erm':
                if i['cp'] == 'scp':
                    erm_scp_detailed_coverage_ver.append(i['test_detailed_coverage'])
                    erm_scp_detailed_size_ver.append(i['test_detailed_size'])
                if i['cp'] == 'wc-scp':
                    erm_wcscp_detailed_coverage_ver.append(i['test_detailed_coverage'])
                    erm_wcscp_detailed_size_ver.append(i['test_detailed_size'])
                if i['cp'] == 'iw-scp':
                    erm_iwscp_detailed_coverage_ver.append(i['test_detailed_coverage'])
                    erm_iwscp_detailed_size_ver.append(i['test_detailed_size'])
            if i['opt'] == 'qr':
                cqr_detailed_coverage_ver.append(i['test_detailed_coverage'])
                cqr_detailed_size_ver.append(i['test_detailed_size'])

        wrcp_overall_gap.append(wrcp_overall_gap_ver)
        wrcp_overall_size.append(wrcp_overall_size_ver)
        wrcp_detailed_coverage.append(wrcp_detailed_coverage_ver)
        wrcp_detailed_size.append(wrcp_detailed_size_ver)

        erm_scp_detailed_coverage.append(erm_scp_detailed_coverage_ver)
        erm_scp_detailed_size.append(erm_scp_detailed_size_ver)

        erm_wcscp_detailed_coverage.append(erm_wcscp_detailed_coverage_ver)
        erm_wcscp_detailed_size.append(erm_wcscp_detailed_size_ver)

        erm_iwscp_detailed_coverage.append(erm_iwscp_detailed_coverage_ver)
        erm_iwscp_detailed_size.append(erm_iwscp_detailed_size_ver)

        cqr_detailed_coverage.append(cqr_detailed_coverage_ver)
        cqr_detailed_size.append(cqr_detailed_size_ver)

    erm_scp_detailed_coverage = np.concatenate([erm_scp_detailed_coverage[i][0] for i in range(len(versions))])
    erm_scp_detailed_size = np.concatenate([erm_scp_detailed_size[i][0] for i in range(len(versions))])
    erm_wcscp_detailed_coverage = np.concatenate([erm_wcscp_detailed_coverage[i][0] for i in range(len(versions))])
    erm_wcscp_detailed_size = np.concatenate([erm_wcscp_detailed_size[i][0] for i in range(len(versions))])
    erm_iwscp_detailed_coverage = np.concatenate([erm_iwscp_detailed_coverage[i][0] for i in range(len(versions))])
    erm_iwscp_detailed_size = np.concatenate([erm_iwscp_detailed_size[i][0] for i in range(len(versions))])
    cqr_detailed_coverage = np.concatenate([cqr_detailed_coverage[i][0] for i in range(len(versions))])
    cqr_detailed_size = np.concatenate([cqr_detailed_size[i][0] for i in range(len(versions))])

    wrcp_weights = []
    for i in RESULT:
        if i['opt'] == 'wr':
            wrcp_weights.append(i['weight'])

    wrcp_detailed_coverage_by_weight = []
    wrcp_detailed_size_by_weight = []

    for w in range(len(wrcp_weights)):
        wrcp_detailed_coverage_by_weight.append(
            np.concatenate([wrcp_detailed_coverage[i][w] for i in range(len(versions))]))
        wrcp_detailed_size_by_weight.append(np.concatenate([wrcp_detailed_size[i][w] for i in range(len(versions))]))

    COVERAGE[idx]['scp'] = erm_scp_detailed_coverage
    COVERAGE[idx]['iw'] = erm_iwscp_detailed_coverage
    COVERAGE[idx]['cqr'] = cqr_detailed_coverage
    COVERAGE[idx]['wc'] = erm_wcscp_detailed_coverage
    COVERAGE[idx]['wrcp'] = wrcp_detailed_coverage_by_weight
    COVERAGE[idx]['wrcp weights'] = wrcp_weights

    SIZE[idx]['scp'] = erm_scp_detailed_size
    SIZE[idx]['iw'] = erm_iwscp_detailed_size
    SIZE[idx]['cqr'] = cqr_detailed_size
    SIZE[idx]['wc'] = erm_wcscp_detailed_size
    SIZE[idx]['wrcp'] = wrcp_detailed_size_by_weight
    SIZE[idx]['wrcp weights'] = wrcp_weights


# raw for alpha, column for dataset
selected_weight_list = [[9,   11, 9, 8, 13, 20],
                        [4.5, 9, 9, 6, 8, 20],
                        [3,   5, 5, 5, 8, 13],
                        [3,   5, 5, 5, 8, 13],
                        [3,   5, 3, 5, 8, 13],
                        [3,   3, 3, 5, 8, 10],
                        [2,   2, 2, 5, 8, 10],
                        [2,   2, 2, 5, 5, 10],
                        [2,   1, 1, 5, 2, 6]]

Size_reduction = []
Coverage_wrcp = []
Coverage_desired = []

for a, alpha in enumerate(alpha_list):

    box_coverage = []
    box_size = []
    for idx, dataset in enumerate(dataset_list):
        selected_weight = selected_weight_list[a][idx]
        pos = COVERAGE[idx]['wrcp weights'].index(selected_weight)
        box_coverage.append([COVERAGE[idx]['scp'][:, a], COVERAGE[idx]['iw'][:, a], COVERAGE[idx]['cqr'][:, a],
                             COVERAGE[idx]['wc'][:, a], COVERAGE[idx]['wrcp'][pos][:, a]])
        box_size.append([SIZE[idx]['scp'][:, a], SIZE[idx]['iw'][:, a], SIZE[idx]['cqr'][:, a],
                         SIZE[idx]['wc'][:, a], SIZE[idx]['wrcp'][pos][:, a]])
        Size_reduction.append(np.mean(SIZE[idx]['wrcp'][pos][:, a] / SIZE[idx]['wc'][:, a]))
        Coverage_wrcp.append(COVERAGE[idx]['wrcp'][pos][:, a])
        Coverage_desired.append((1 - alpha) * np.ones(len(COVERAGE[idx]['wrcp'][pos][:, a])))

    colors = ['peachpuff', 'lightgreen', 'lemonchiffon', 'lightpink', 'lightblue']
    labels = ['Vanilla CP', 'IW-CP', 'CQR', 'WC-CP', 'WR-CP']
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
