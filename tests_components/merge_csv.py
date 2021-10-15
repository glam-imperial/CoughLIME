import pandas as pd
import pixelFlipping
import csv
import numpy as np


def merge():
    a = pd.read_csv("../old_evals/Spectral /10_06_pixelf/part1/pixel_flipping.csv")
    b = pd.read_csv("../old_evals/Spectral /10_06_pixelf/part2/pixel_flipping.csv")
    print(a.shape, b.shape)
    merged = pd.concat([a, b])
    print(merged.shape)
    merged.to_csv("../old_evals/Spectral /10_06_pixelf/pixel_flipping.csv", index=False)


def evaluate(comps):
    read_file = open('../old_evals/Spectral /10_06_pixelf/part1/pixel_flipping.csv', 'r')
    csv_reader = csv.reader(read_file)
    _ = next(csv_reader)
    number_files = 0
    true_morf = [0] * len(comps)
    true_rand = [0] * len(comps)

    for row in csv_reader:
        if row[2] == 'morf':
            prediction_whole = np.rint(float(row[1]))
            for i in range(len(comps)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_morf[i] += 1
        else:
            prediction_whole = np.rint(float(row[1]))
            number_files += 1
            for i in range(len(comps)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_rand[i] += 1
    read_file.close()

    read_file = open('../old_evals/Spectral /10_06_pixelf/part2/pixel_flipping.csv', 'r')
    csv_reader = csv.reader(read_file)
    _ = next(csv_reader)

    for row in csv_reader:
        if row[2] == 'morf':
            prediction_whole = np.rint(float(row[1]))
            for i in range(len(comps)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_morf[i] += 1
        else:
            prediction_whole = np.rint(float(row[1]))
            number_files += 1
            for i in range(len(comps)):
                if np.rint(float(row[3 + i])) == prediction_whole:
                    true_rand[i] += 1
    read_file.close()

    for index, removed in enumerate(comps):
        path_save_summary = f"../old_evals/Spectral /10_06_pixelf/{removed}_removed_components.txt"
        with open(path_save_summary, 'w') as summary:
            summary.write(f"Number samples: {number_files}")
            summary.write("\n")
            summary.write(f"Number true explanations: {true_morf}")
            summary.write("\n")
            summary.write(f"Number true random predictions: {true_rand}")
            summary.write("\n")
            summary.write(f"Percentage of true explanations: {float(true_morf[index]) / float(number_files)}")
            summary.write("\n")
            summary.write(f"Percentage of random true predictions: {float(true_rand[index]) / float(number_files)}")

if __name__ == "__main__":
    # merge()
    comps = list(range(7))
    evaluate(comps)
    # pixelFlipping.evaluate_data(comps, "../old_evals/Spectral /10_06_pixelf")


