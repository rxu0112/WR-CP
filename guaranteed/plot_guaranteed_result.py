import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import os

def find_position_unsorted(arr, value):
    best_index = -1
    best_value = None
    for i, elem in enumerate(arr):
        if elem < value and (best_value is None or elem > best_value):
            best_value = elem
            best_index = i
    return best_index if best_index != -1 else None

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_list = ['airfoil', 'pemsd4', 'pemsd8', 'seattle', 'states', 'japan']
versions = range(1, 11)
alpha_list = [0.1]

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

    erm_wccp_detailed_coverage = []
    erm_wccp_detailed_size = []

    for v in versions:
        wrcp_path = os.path.join(project_path, "main/main_result", dataset, f"v{v}")
        RESULT_WRCP = pickle.load(open(wrcp_path, "rb"))
        wccp_path = os.path.join(project_path, "guaranteed/guaranteed_result", dataset, f"wc_v{v}")
        RESULT_WC = pickle.load(open(wccp_path, "rb"))

        wrcp_overall_gap_ver = []
        wrcp_overall_size_ver = []
        wrcp_detailed_coverage_ver = []
        wrcp_detailed_size_ver = []

        erm_wccp_detailed_coverage_ver = []
        erm_wccp_detailed_size_ver = []

        for i in RESULT_WRCP:
            if i['opt'] == 'wr':
                wrcp_overall_gap_ver.append(i['test_overall_gap'])
                wrcp_overall_size_ver.append(i['test_overall_size'])
                wrcp_detailed_coverage_ver.append(i['test_detailed_coverage'])  # average over envs
                wrcp_detailed_size_ver.append(i['test_detailed_size'])

        for i in RESULT_WC:
            if i['opt'] == 'erm':
                if i['cp'] == 'wc-scp':
                    erm_wccp_detailed_coverage_ver.append(i['test_detailed_coverage'])
                    erm_wccp_detailed_size_ver.append(i['test_detailed_size'])

        wrcp_overall_gap.append(wrcp_overall_gap_ver)
        wrcp_overall_size.append(wrcp_overall_size_ver)
        wrcp_detailed_coverage.append(wrcp_detailed_coverage_ver)
        wrcp_detailed_size.append(wrcp_detailed_size_ver)

        erm_wccp_detailed_coverage.append(erm_wccp_detailed_coverage_ver)
        erm_wccp_detailed_size.append(erm_wccp_detailed_size_ver)

    erm_wccp_detailed_coverage = np.concatenate([erm_wccp_detailed_coverage[i][0] for i in range(len(versions))])
    erm_wccp_detailed_size = np.concatenate([erm_wccp_detailed_size[i][0] for i in range(len(versions))])

    wrcp_weights = []
    for i in RESULT_WRCP:
        if i['opt'] == 'wr':
            wrcp_weights.append(i['weight'])

    wrcp_detailed_coverage_by_weight = []
    wrcp_detailed_size_by_weight = []

    for w in range(len(wrcp_weights)):
        wrcp_detailed_coverage_by_weight.append(
            np.concatenate([wrcp_detailed_coverage[i][w] for i in range(len(versions))]))
        wrcp_detailed_size_by_weight.append(np.concatenate([wrcp_detailed_size[i][w] for i in range(len(versions))]))

    COVERAGE[idx]['wc'] = erm_wccp_detailed_coverage
    COVERAGE[idx]['wrcp'] = wrcp_detailed_coverage_by_weight
    COVERAGE[idx]['wrcp weights'] = wrcp_weights

    SIZE[idx]['wc'] = erm_wccp_detailed_size
    SIZE[idx]['wrcp'] = wrcp_detailed_size_by_weight
    SIZE[idx]['wrcp weights'] = wrcp_weights


# raw for alpha, column for dataset
selected_weight_list = [[1, 1, 1, 1, 1, 1]]

Size_reduction = []
Coverage_wrcp = []

for idx, dataset in enumerate(dataset_list):
    grouped_results = {}
    for v in versions:
        bound_path = os.path.join(project_path, "guaranteed/guaranteed_result", dataset, f"v{v}")
        RESULT_Bound = pickle.load(open(bound_path, "rb"))
        for i in RESULT_Bound:
            if float(i['weight']) not in grouped_results:
                grouped_results[float(i['weight'])] = []
            grouped_results[float(i['weight'])].append(i['gap_bound_list'])

    for i in grouped_results:
        grouped_results[i] = np.max(np.array(grouped_results[i]), axis=0)
    COVERAGE[idx]['wrcp_lower_bound'] = grouped_results

for a, alpha in enumerate(alpha_list):

    box_coverage = []
    box_size = []
    guaranteed_confidence = []
    for idx, dataset in enumerate(dataset_list):
        selected_weight = selected_weight_list[a][idx]
        pos_w = COVERAGE[idx]['wrcp weights'].index(selected_weight)
        for w in COVERAGE[idx]['wrcp_lower_bound']:
            if w == selected_weight:
                lower_bound = COVERAGE[idx]['wrcp_lower_bound'][w][a]

        guaranteed_alpha = alpha + max(lower_bound)
        guaranteed_confidence.append([1 - guaranteed_alpha])

        box_coverage.append([COVERAGE[idx]['wc'][:, a], COVERAGE[idx]['wrcp'][pos_w][:, a]])
        box_size.append([SIZE[idx]['wc'][:, a], SIZE[idx]['wrcp'][pos_w][:, a]])
        Size_reduction.append(np.mean(SIZE[idx]['wrcp'][pos_w][:, a] / SIZE[idx]['wc'][:, a]))
        Coverage_wrcp.append(COVERAGE[idx]['wrcp'][pos_w][:, a])

    colors = ['lightpink', 'lightblue']
    labels = ['WC-CP', 'WR-CP']
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    titles = ['Airfoil', 'PeMSD4', 'PeMSD8', 'Seattle', 'US', 'Japan']

    fig, axs = plt.subplots(2, 6, figsize=(13.8, 3.6))
    fig.subplots_adjust(left=0.05, bottom=0.14, right=0.995, top=0.905, hspace=0.083, wspace=0.31)

    for i in range(len(dataset_list)):
        box = axs[0, i].boxplot(box_coverage[i], showfliers=False, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Add a reference horizontal line
        axs[0, i].axhline(y=guaranteed_confidence[i][0], color='r', linestyle='--')
        axs[0, i].title.set_text(titles[i])
        axs[0, i].title.set_fontsize(12)
        axs[0, i].set_xticks([])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_ylim(0.2*guaranteed_confidence[i][0], 1.05)


        if i == 0:
            axs[0, i].set_ylabel('Coverage', fontsize=12)
    line_legend = mlines.Line2D([], [], color='red', linestyle='--', label='Lowest Guaranteed Coverage')

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
            axs[1, i].set_ylabel('Pred. Set Size', fontsize=12)
    fig.legend(handles=legend_patches + [line_legend], loc='lower center', ncol=6, prop={'size': 12})

    plt.show()

