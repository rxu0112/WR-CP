import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_list = ['airfoil', 'pemsd4', 'pemsd8', 'seattle', 'states', 'japan']
versions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
alpha_list = [0.2]

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

    wrcp_uw_overall_gap = []  # (#weight,1)
    wrcp_uw_overall_size = []  # (#weight,1)
    wrcp_uw_detailed_coverage = []  # (#weight,#alpha)
    wrcp_uw_detailed_size = []  # (#weight,#alpha)

    for v in versions:
        RESULT = pickle.load(open(project_path + "/main/main_result/" + dataset + "/v" + str(v), "rb"))
        wrcp_overall_gap_ver = []
        wrcp_overall_size_ver = []
        wrcp_detailed_coverage_ver = []
        wrcp_detailed_size_ver = []

        for i in RESULT:
            if i['opt'] == 'wr':
                wrcp_overall_gap_ver.append(i['test_overall_gap'])
                wrcp_overall_size_ver.append(i['test_overall_size'])
                wrcp_detailed_coverage_ver.append(i['test_detailed_coverage'])  # average over envs
                wrcp_detailed_size_ver.append(i['test_detailed_size'])

        wrcp_overall_gap.append(wrcp_overall_gap_ver)
        wrcp_overall_size.append(wrcp_overall_size_ver)
        wrcp_detailed_coverage.append(wrcp_detailed_coverage_ver)
        wrcp_detailed_size.append(wrcp_detailed_size_ver)

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

    COVERAGE[idx]['wrcp'] = wrcp_detailed_coverage_by_weight
    COVERAGE[idx]['wrcp weights'] = wrcp_weights

    SIZE[idx]['wrcp'] = wrcp_detailed_size_by_weight
    SIZE[idx]['wrcp weights'] = wrcp_weights

    for v in versions:
        RESULT_UW = pickle.load(open(project_path + "/unweighted/unweighted_result/" + dataset + "/v" + str(v), "rb"))
        wrcp_uw_overall_gap_ver = []
        wrcp_uw_overall_size_ver = []
        wrcp_uw_detailed_coverage_ver = []
        wrcp_uw_detailed_size_ver = []

        for i in RESULT_UW:
            if i['opt'] == 'wr_uw':
                wrcp_uw_overall_gap_ver.append(i['test_overall_gap'])
                wrcp_uw_overall_size_ver.append(i['test_overall_size'])
                wrcp_uw_detailed_coverage_ver.append(i['test_detailed_coverage'])  # average over envs
                wrcp_uw_detailed_size_ver.append(i['test_detailed_size'])

        wrcp_uw_overall_gap.append(wrcp_uw_overall_gap_ver)
        wrcp_uw_overall_size.append(wrcp_uw_overall_size_ver)
        wrcp_uw_detailed_coverage.append(wrcp_uw_detailed_coverage_ver)
        wrcp_uw_detailed_size.append(wrcp_uw_detailed_size_ver)

    wrcp_uw_weights = []
    for i in RESULT_UW:
        if i['opt'] == 'wr_uw':
            wrcp_uw_weights.append(i['weight'])

    wrcp_uw_detailed_coverage_by_weight = []
    wrcp_uw_detailed_size_by_weight = []

    for w in range(len(wrcp_uw_weights)):
        wrcp_uw_detailed_coverage_by_weight.append(
            np.concatenate([wrcp_uw_detailed_coverage[i][w] for i in range(len(versions))]))
        wrcp_uw_detailed_size_by_weight.append(np.concatenate([wrcp_uw_detailed_size[i][w] for i in range(len(versions))]))

    COVERAGE[idx]['wrcp_uw'] = wrcp_uw_detailed_coverage_by_weight
    COVERAGE[idx]['wrcp_uw weights'] = wrcp_uw_weights

    SIZE[idx]['wrcp_uw'] = wrcp_uw_detailed_size_by_weight
    SIZE[idx]['wrcp_uw weights'] = wrcp_uw_weights

# raw for alpha, column for dataset
selected_weight_list = [4.5, 9, 9, 6, 8, 20]

alpha = 0.2

box_coverage = []
box_size = []
Coverage_desired = []
Coverage_wrcp = []
Coverage_wrcp_uw = []
Size_reduction = []
for idx, dataset in enumerate(dataset_list):
    selected_weight = selected_weight_list[idx]
    pos_wrcp = COVERAGE[idx]['wrcp weights'].index(selected_weight)
    pos_wrcp_uw = COVERAGE[idx]['wrcp_uw weights'].index(selected_weight)

    box_coverage.append([COVERAGE[idx]['wrcp'][pos_wrcp][:, 1], COVERAGE[idx]['wrcp_uw'][pos_wrcp_uw][:, 1]])
    box_size.append([SIZE[idx]['wrcp'][pos_wrcp][:, 1], SIZE[idx]['wrcp_uw'][pos_wrcp_uw][:, 1]])
    Coverage_desired.append((1 - alpha) * np.ones(len(COVERAGE[idx]['wrcp'][pos_wrcp][:, 1])))
    Coverage_wrcp.append(COVERAGE[idx]['wrcp'][pos_wrcp][:, 1])
    Coverage_wrcp_uw.append(COVERAGE[idx]['wrcp_uw'][pos_wrcp_uw][:, 1])
    Size_reduction.append(np.mean(SIZE[idx]['wrcp'][pos_wrcp][:, 1] / SIZE[idx]['wrcp_uw'][pos_wrcp_uw][:, 1]))

colors = ['lightblue', 'gray']
labels = ['WR-CP', 'WR-CP(uw)']
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
    axs[0, i].set_ylim(0.4*(1-alpha), 1.05)


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

gap_wrcp = []
for i in range(6):
    gap_wrcp.append(np.mean(np.abs(Coverage_desired[i]-Coverage_wrcp[i])))

gap_wrcp_uw = []
for i in range(6):
    gap_wrcp_uw.append(np.mean(np.abs(Coverage_desired[i]-Coverage_wrcp_uw[i])))
print(np.mean(gap_wrcp))
print(np.mean(gap_wrcp_uw))
print(1-np.mean(Size_reduction))