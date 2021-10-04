import numpy as np
import re


def extract_data(summary_path):
    """
    extracts the data from one text file generated during pixel flipping
    :param summary_path: path to text file
    :return: percentage of correct predictions for morf, percentage of correct random predictions (baseline)
    """
    summary_file = open(summary_path, 'r')
    lines = summary_file.readlines()
    string_percentage_exp = lines[3]
    string_percentage_rand = lines[4]
    pattern = r"0\.\d*"  # regex pattern to match the percentage
    percentage_rand = re.search(pattern, string_percentage_rand).group()
    percentage_exp = re.search(pattern, string_percentage_exp).group()
    return percentage_exp, percentage_rand


def calculate_dauc(comps, data_dir):
    """
    calculates the delta area under curve to quantify the performance of CoughLIME
    :param comps: 1d list, contains all numbers/percentages of removed components from the evaluation to quantify
    :param data_dir: path to where the evaluation text files are stored
    :return: float, the calculated delta AUC
    """
    morfs = np.zeros(len(comps))
    rands = np.zeros(len(comps))
    for i, c in enumerate(comps):
        summary_path = f"{data_dir}/{c}_removed_components.txt"  # TODO:adapt
        morfs[i], rands[i] = extract_data(summary_path)
    area_morf = calculate_area(morfs, comps)
    area_rand = calculate_area(rands, comps)
    dauc = abs(area_rand - area_morf)
    print("Delta Area Under Curve:", dauc)
    return dauc


def calculate_area(percentages, comps):
    """
    sums up the integrals of the different parts of the curve to get the entire area under one curve
    :param percentages: np.arrau(n,), y values of the function
    :param comps: 1d list, x values of function, components used for pixel flipping evaluation
    :return: float: summed up integral, area under the curve
    """
    # need to redistribute comps s.t. they are normalized between 0 and 1 for comparison
    normalized_c = np.array(comps)/max(comps)
    area = 0
    for i in range(1, len(comps)):
        area += integral(normalized_c[i], normalized_c[i - 1], percentages[i], percentages[i - 1])
    return area


def integral(x_1, x_0, y_1, y_0):
    '''
    calculates the integral of a linear function f(x) = (y_1 - y_0)/(x_1 - x_0) * x + y_0 from x_0 to x_1
    :return: integral as float
    '''
    # integral of linear function ax + b
    a = (y_1 - y_0) / (x_1 - x_0)
    b = y_0
    length = (x_1 - x_0)  # length over which to calculate the area
    result = (a / 2) * (length ** 2) + b * length
    return result


def main(components, path):
    return calculate_dauc(components, path)


if __name__ == '__main__':
    # comp = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    comp = [0, 1, 2, 3, 4, 5, 6]
    print(calculate_dauc(comp, './old_evals/Temporal/09_13_pixel_flipping_temporal/eval'))  # TODO: adapt
