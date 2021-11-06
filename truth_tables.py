import pandas as pd
import numpy as np
import csv


def main_truth_tables(components, path_file, path_save):
    """
    main function calculating and saving the number of true and false positives and true and false negatives
    :param components: list of different k components to evaluate at
    :param path_file: path to the .csv file containing the evaluation results
    :param path_save: path to save the .txt files containing the truth table values
    """
    # get values of true positives, false positives, true negatives and false negatives
    # iterate over components, print to output text file
    read_file = open(f'{path_file}', 'r')
    csv_reader = csv.reader(read_file, delimiter=";")  # TODO: adapt delimiter
    _ = next(csv_reader)
    true_pos_exp = [0] * len(components)
    true_pos_rand = [0] * len(components)
    true_neg_exp = [0] * len(components)
    true_neg_rand = [0] * len(components)
    false_neg_exp = [0] * len(components)
    false_neg_rand = [0] * len(components)
    false_pos_exp = [0] * len(components)
    false_pos_rand = [0] * len(components)

    for row in csv_reader:
        if row[2] == 'morf':
            prediction_whole = np.rint(float(row[1]))
            if prediction_whole == 1:
                # file is positive
                for i in range(len(components)):
                    if np.rint(float(row[3 + i])) == 1:
                        true_pos_exp[i] += 1
                    else:
                        false_neg_exp[i] += 1
            else:
                for i in range(len(components)):
                    if np.rint(float(row[3 + i])) == 0:
                        true_neg_exp[i] += 1
                    else:
                        false_pos_exp[i] += 1
        else:
            prediction_whole = np.rint(float(row[1]))
            if prediction_whole == 1:
                # file is positive
                for i in range(len(components)):
                    if np.rint(float(row[3 + i])) == 1:
                        true_pos_rand[i] += 1
                    else:
                        false_neg_rand[i] += 1
            else:
                for i in range(len(components)):
                    if np.rint(float(row[3 + i])) == 0:
                        true_neg_rand[i] += 1
                    else:
                        false_pos_rand[i] += 1
    read_file.close()

    for index, c in enumerate(components):
        # save these to txt file for further processing
        path_summary = f"{path_save}/{c}_components.txt"
        with open(path_summary, 'w') as summary:
            summary.write(f"True positive explanations: {true_pos_exp[index]}")
            summary.write("\n")
            summary.write(f"True positive random: {true_pos_rand[index]}")
            summary.write("\n")
            summary.write(f"False positive explanations: {false_pos_exp[index]}")
            summary.write("\n")
            summary.write(f"False positive random: {false_pos_rand[index]}")
            summary.write("\n")
            summary.write(f"True negative explanations: {true_neg_exp[index]}")
            summary.write("\n")
            summary.write(f"True negative random: {true_neg_rand[index]}")
            summary.write("\n")
            summary.write(f"False negative explanations: {false_neg_exp[index]}")
            summary.write("\n")
            summary.write(f"False negative random: {false_neg_rand[index]}")
            summary.write("\n")


if __name__ == "__main__":
    comps = list(range(7))#[0, 0.1, 0.25, 0.5, 0.75, 0.9] #list(range(7))
    path_file = './old_evals/Spectral /fold1/output_run_1/pixel_flipping.csv'
    path_save = './truth_tables/s_dicova/run_1'
    main_truth_tables(comps, path_file, path_save)
