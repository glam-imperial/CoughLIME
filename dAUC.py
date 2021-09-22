import numpy as np
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


def calculate_dauc(comps, data_dir):
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
    # sum up integrals of comp i - 1 to comp i
    # need to redistribute comps s.t. they are normalized between 0 and 1 for comparison
    normalized_c = np.array(comps)/max(comps)
    area = 0
    for i in range(1, len(comps)):
        area += integral(normalized_c[i], normalized_c[i - 1], percentages[i], percentages[i - 1])
    return area


def integral(x_1, x_0, y_1, y_0):
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
    print(calculate_dauc(comp, './old_evals/Temporal/09_13_pixel_flipping_temporal/eval'))
