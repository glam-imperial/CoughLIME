import numpy as np
import matplotlib.pyplot as plt
import re
import statistics
import os
import matplotlib.ticker as mtick
import dAUC


def extract_data(summary_path):
    """
    extracts the needed percentages from a text file generated during the evaluation
    :param summary_path: path to text file
    :return: percentage of morf leading to same prediction, percentage of random components leading to same predictions
    """
    summary_file = open(summary_path, 'r')
    lines = summary_file.readlines()
    string_percentage_exp = lines[3]
    string_percentage_rand = lines[4]
    pattern = r"0\.\d*"
    percentage_rand = re.search(pattern, string_percentage_rand).group()
    percentage_exp = re.search(pattern, string_percentage_exp).group()
    return percentage_exp, percentage_rand


def make_graph(comps, values, save_path, plot_dev=False, dev=None, mode='random', dauc=None):
    """
    function that makes a graph summarizing the evaluation
    :param comps: list of components that were included -> for x axis
    :param values: list of values -> for y axis of graph
    :param save_path: path to save the generated figure
    :param plot_dev: bool, whether to also plot the standard deviation as errorbar
    :param dev: if not None: list of std devs to plot
    :param mode: can be 'random' or 'exp', if random: plot std dev for random prediction, if exp: plot std dev for morf
    :param dauc: float or None: delta area under curve, if None: don't plot the delta AUC
    :return: nothing, saves and shows generated figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.9)
    for index, c in enumerate(comps):
        if plot_dev:
            if mode == 'exp':
                print(dev[index])
                plt.errorbar(c, values[0, index], dev[index], marker='.', color='r')
            elif mode == 'random':
                print(dev[index])
                plt.errorbar(c, values[1, index], dev[index], marker='.', color='b')
        plt.plot(c, values[1, index], 'b.')
        label = "{:.2f}".format(values[1, index])
        plt.annotate(label,
                     (c, values[1, index]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')
        plt.plot(c, values[0, index], 'r.')
        label = "{:.2f}".format(values[0, index])
        plt.annotate(label,
                     (c, values[0, index]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')
    plt.plot(comps, values[1, :], 'b', label='Random components flipped')
    plt.plot(comps, values[0, :], 'r', label='Most relevant components flipped first')
    if dauc is not None:
        # need to color the area in between the curves and add the dauc value to the plot
        lbl = f"Delta-Area Under Curve = {round(dauc, 4)}"
        plt.fill_between(comps, values[0, :], values[1, :], color='mediumpurple', alpha=0.5, label=lbl)
    plt.title("Loudness: averaged percentage of predictions leading\nto same class for k % flipped components")  # TODO: adapt
    plt.xticks(comps)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel("Percentage of flipped components")
    plt.ylabel("Percentage of same predictions")
    plt.legend()
    plt.savefig(f'{save_path}/summary_plot_average.png')
    plt.show()


def calculate_std_var(exps, comp, path):
    """
    calculate std deviation and variance and saves them to an overview text file
    :param exps: 2d array of values over various runs
    :param comp: list of number of removed components for which the results have been generated
    :param path: path to save the overview text file
    :return: std deviations
    """
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


def make_single_graph(components, path_text_files, dauc=True):
    """
    wrapper function that calls the subordinate functions to make a graph
    :param components: list of number of removed components for which the evaluation was conducted
    :param path_text_files: path to the results of the evaluation
    :param dauc: bool, whether to include the delta area under curve metric in the graph
    """
    percentages = np.zeros((2, len(components)))
    for i, comp in enumerate(components):
        summary_path = f"{path_text_files}/{comp}_removed_components.txt"  # TODO:adapt
        exp, rand = extract_data(summary_path)
        percentages[0, i] = exp
        percentages[1, i] = rand
    if dauc:
        dauc = dAUC.main(components, path_text_files)
        make_graph(components, percentages, path_text_files, dauc=dauc)
    else:
        make_graph(components, percentages, path_text_files, dauc=None)


def get_significance_metrics(exp, rand, comps, path_to_save):
    """
    wrapper function to make graph for an evaluation that includes various runs
    :param exp: 2d list, explanation scores over various runs
    :param rand: 2d list, random scores over various runs
    :param comps: list of number of removed components for which the evaluation was conducted
    :param path_to_save: path for saving the results
    """
    average_exp = np.mean(exp, axis=1)
    average_rand = np.mean(rand, axis=1)

    # make graph for average
    average_values = np.stack((average_exp, average_rand))
    stdev = calculate_std_var(exp, comps, path_to_save)
    make_graph(comps, average_values, path_to_save, plot_dev=True, dev=stdev, mode='exp')


def best_performing(exps, rands, comps, path_to_save, dauc=False):
    """
    wrapper function in case the best performing morf model should be selected and averaged performance for random
    component selection
    :param exps: 2d list, explanation scores over various runs
    :param rands: 2d list, random scores over various runs
    :param comps: list of number of removed components for which the evaluation was conducted
    :param path_to_save: path for saving the results
    :param dauc: bool, whether to include delta AUC in graph
    """
    average_rand = np.mean(rands, axis=1)
    sums_models = np.sum(exps, axis=0)
    best_exp = exps[:, np.argmin(sums_models)]  # TODO: adapt to pixel flipping/quantitative
    # check dimensions, compare with previous versions
    values = np.stack((best_exp, average_rand))
    stdev = calculate_std_var(rands, comps, path_to_save)
    if dauc:
        area_morf = dAUC.calculate_area(best_exp, comps)
        area_rand = dAUC.calculate_area(average_rand, comps)
        dauc = abs(area_rand - area_morf)
        print("Delta Area Under Curve:", dauc)
    make_graph(comps, values, path_to_save, plot_dev=True, dev=stdev, mode='random', dauc=dauc)


def significance_analysis(comps, path, stddev='random', dauc=False):
    """
    wrapper function for results over various runs
    :param comps: list of number of removed components for which the evaluation was conducted
    :param path: path to results of evaluation
    :param stddev: can be 'random' or 'exp', if random: plot std dev for random prediction, if exp: plot std dev for morf
    :param dauc: bool, whether to include delta AUC in graph
    """
    runs = 5
    explanations = np.zeros((len(comps), runs))
    randoms = np.zeros((len(comps), runs))
    for run in range(runs):
        for comp_index, comp in enumerate(comps):
            summary_path = f"{path}/output_run_{run}/{comp}_removed_components.txt"
            explanations[comp_index, run], randoms[comp_index, run] = extract_data(summary_path)
    if stddev == 'exp':
        get_significance_metrics(explanations, randoms, comps, path)
    elif stddev == 'random':
        best_performing(explanations, randoms, comps, path, dauc)


if __name__ == "__main__":
    components = [0, 0.1, 0.25, 0.5, 0.75, 0.9]  # TODO: adapt
    path_text_files = '/Users/anne/PycharmProjects/LIME_cough/old_evals/Loudness/09_22_pixel_flipping_5_runs/eval'  # TODO: adapt
    # make_single_graph(components, path_text_files, dauc=True)  # TODO: adapt
    significance_analysis(components, path_text_files, stddev='random', dauc=True)
    print('All done :) ')
