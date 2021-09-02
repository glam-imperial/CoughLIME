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


def make_graph(comps, values, save_path, plot_dev=False, dev=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.9)
    for index, c in enumerate(comps):
        if plot_dev:
            print(dev[index])
            plt.errorbar(c, values[0, index], dev[index], marker='.', color='r')
        else:
            plt.plot(c, values[0, index], 'r.')
        label = "{:.2f}".format(values[0, index])
        plt.annotate(label,
                     (c, values[0, index]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')
        plt.plot(c, values[1, index], 'b.')
        label = "{:.2f}".format(values[1, index])
        plt.annotate(label,
                     (c, values[1, index]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')

    plt.plot(comps, values[0, :], 'r', label='Most relevant first flipped')
    plt.plot(comps, values[1, :], 'b', label='Random components flipped')
    plt.title("Temporal: percentage of predictions leading to\nsame class for k out of 7 flipped components")  # TODO: adapt
    plt.xticks(comps)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel("Number of components")
    plt.ylabel("Percentage of same predictions")
    plt.legend()
    plt.savefig(f'{save_path}/summary_plot_average.png')
    plt.show()


def calculate_std_var(exps, comp, path):
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
            summary.write(f'For {c} components, the standard deviation of the generated explanations is {current_dev} and the variance is {current_var}\n')
        summary.write(f"Averaged standard deviation over all components: {np.mean(deviations)}\n")
        summary.write(f"Averaged variance over all components: {np.mean(variances)}\n")
    return deviations


def make_single_graph(components, path_text_files):
    percentages = np.zeros((2, len(components)))
    for i, comp in enumerate(components):
        summary_path = f"{path_text_files}/{comp}_removed_components.txt" # TODO:adapt
        exp, rand = extract_data(summary_path)
        percentages[0, i] = exp
        percentages[1, i] = rand

    make_graph(components, percentages, path_text_files)


def get_significance_metrics(exp, rand, comps, path_to_save):
    average_exp = np.mean(exp, axis=1)
    average_rand = np.mean(rand, axis=1)

    # make graph for average
    average_values = np.stack((average_exp, average_rand))
    stdev = calculate_std_var(exp, comps, path_to_save)
    make_graph(comps, average_values, path_to_save, plot_dev=True, dev=stdev)


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
    components = [1, 2, 3, 4, 5, 6]
    path_text_files = '/Users/anne/PycharmProjects/LIME_cough/eval/'  # TODO: adapt
    make_single_graph(components, path_text_files)
    # significance_analysis(components, path_text_files)
    print('All done :) ')
