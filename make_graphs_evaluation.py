import numpy as np
import matplotlib.pyplot as plt
import re

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
    for index, c in enumerate(comps):
        plt.plot(c, values[0, index], 'ro')
        plt.plot(c, values[1, index], 'bo')
    plt.plot(comps, values[0, :], 'r', label='Explanations')
    plt.plot(comps, values[1, :], 'b', label='Random components')
    plt.title("MFCC Direct: correct percentage per k out of 7 components")  # TODO: adapt
    plt.xticks(comps)

    plt.xlabel("Number of components")
    plt.ylabel("Percentage of correct predictions")
    plt.legend()
    plt.savefig(f'{save_path}/summary_plot.png')
    print("All done :) ")


if __name__ == "__main__":
    components = [1, 3, 5, 7]
    path_text_files = '/Users/anne/PycharmProjects/LIME_cough/old_eval/quantitative_evaluation_mfcc_direct/'  # TODO: adapt
    percentages = np.zeros((2, len(components)))
    for i, comp in enumerate(components):
        summary_path = f"{path_text_files}/{comp}_components.txt"
        exp, rand =  extract_data(summary_path)
        percentages[0, i] = exp
        percentages[1, i] = rand

    make_graph(components, percentages, path_text_files)