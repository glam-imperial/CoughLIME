import numpy as np
import matplotlib.pyplot as plt
import re
import statistics
import os
import matplotlib.ticker as mtick

def extract_data(summary_path):
    summary_file = open(summary_path, 'r')
    lines = summary_file.readlines()
    string_percentage_exp = lines[3]
    string_percentage_rand = lines[4]
    pattern = r"0\.\d*"
    percentage_rand = re.search(pattern, string_percentage_rand).group()
    percentage_exp = re.search(pattern, string_percentage_exp).group()
    return percentage_exp, percentage_rand


def make_graph(comps, values, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.9)
    for index, c in enumerate(comps):
        plt.plot(c, values[0, index], 'ro')
        plt.plot(c, values[1, index], 'bo')
    plt.plot(comps, values[0, :], 'r', label='Explanations')
    plt.plot(comps, values[1, :], 'b', label='Random components')
    plt.title("Loudness: percentage per k out of 2 components")  # TODO: adapt
    plt.xticks(comps)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel("Number of components")
    plt.ylabel("Percentage of correct predictions")
    plt.legend()
    plt.savefig(f'{save_path}/summary_plot_average.png')
    plt.show()


def make_scattered_graph(average, exps, rands, comp, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.9)
    for run in range(np.shape(exps)[1]):
        for i, c in enumerate(comp):
            plt.plot(c, exps[i, run], 'ro')
            plt.plot(c, rands[i, run], 'bo')
    plt.plot(comp, average[0, :], 'r', label='Explanations')
    plt.plot(comp, average[1, :], 'b', label='Random components')
    plt.title("Temporal: averaged percentage and all points\nper k out of 7 components over 10 runs")  # TODO: adapt
    plt.xticks(comp)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel("Number of components")
    plt.ylabel("Percentage of correct predictions")
    plt.legend()
    plt.savefig(f'{path}/summary_plot_scattered.png')
    plt.subplot_tool()
    plt.show()

    # calculate standard deviation and variance
    deviations = np.zeros((len(comp)))
    variances = np.zeros((len(comp)))
    path_save_summary = f"{path}/statistics.txt"
    with open(path_save_summary, 'w') as summary:
        for index, c in enumerate(comp):
            current_dev = statistics.stdev(exps[index, :])
            current_var = statistics.variance(exps[index, :])
            deviations[index] = current_dev
            variances[index] = current_var
            summary.write(f'For {c} components, the standard deviation is {current_dev} and the variance is {current_var}\n')
        summary.write(f"Averaged standard deviation over all components: {np.mean(deviations)}\n")
        summary.write(f"Averaged variance over all components: {np.mean(variances)}\n")


def make_single_graph(components, path):
    percentages = np.zeros((2, len(components)))
    for i, comp in enumerate(components):
        summary_path = f"{path_text_files}/{comp}_components.txt"
        exp, rand =  extract_data(summary_path)
        percentages[0, i] = exp
        percentages[1, i] = rand

    make_graph(components, percentages, path_text_files)


def get_significance_metrics(exp, rand, comps, path_to_save):
    average_exp = np.mean(exp, axis=1)
    average_rand = np.mean(rand, axis=1)

    # make graph for average
    average_values = np.stack((average_exp, average_rand))
    make_graph(comps, average_values, path_to_save)

    make_scattered_graph(average_values, exp, rand, comps, path_to_save)


def significance_analysis(comps, path):
    runs = 5
    explanations = np.zeros((len(comps), runs))
    randoms = np.zeros((len(comps), runs))
    for run in range(runs):
        for comp_index, comp in enumerate(comps):
            summary_path = f"{path}/output_run_{run}/{comp}_components.txt"
            explanations[comp_index, run], randoms[comp_index, run] = extract_data(summary_path)
    get_significance_metrics(explanations, randoms, comps, path)


if __name__ == "__main__":
    components = [1, 2]  # TODO: adapt
    path_text_files = '/Users/anne/PycharmProjects/LIME_cough/old_evals/loudness_2_comp/quantitative_evaluation/output_run_0'  # TODO: adapt
    make_single_graph(components, path_text_files)
    # significance_analysis(components, path_text_files) # TODO: adapt
    print('All done :) ')
