import numpy as np
import matplotlib.pyplot as plt
import re
import statistics
import os
import matplotlib.ticker as mtick
import dAUC
import make_graphs_evaluation


def get_dauc_values(list_folders, list_numbers, path, comps=None, eval_type=None):
    list_dauc = []
    for index, folder in enumerate(list_folders):
        if eval_type == 'flexible':
            comps = list(range(list_numbers[index]))
        joined_path = f"{path}/{folder}"
        dauc = dAUC.calculate_dauc(comps, joined_path)
        list_dauc.append(dauc)

    if len(list_dauc) != len(list_numbers):
        print('Error, lists have different lengths')

    return list_dauc


def make_dauc_graph(numbers, dauc, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.9)
    plt.plot(numbers, dauc, 'r')
    plt.title("Loudness: achieved dAUC for different thresholds")  # TODO: adapt
    plt.xticks(numbers)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel("Threshold in dB for loudness decomposition")  # TODO: adapt
    plt.ylabel("dAUC (in %)")
    plt.savefig(f'{save_path}/summary_plot.png')
    plt.show()


def main(path_files, list_numbers, list_folders, comps=None, eval_type=None):
    list_dauc = get_dauc_values(list_folders, list_numbers, path_files, comps, eval_type)
    make_dauc_graph(list_numbers, list_dauc, path_files)


if __name__ == '__main__':
    # TODO: adapt
    numbers = [45, 55, 65, 75, 85, 95]
    comps = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    folders = []
    path = './hyperparams/loudness/10_13_threshold_min=0'
    for n in numbers:
        folders.append(f'threshold_{n}')

    main(path, numbers, folders, comps)
