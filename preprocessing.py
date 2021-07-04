"""
This file is used to implement preprocessing.
"""


# import libraries
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # load all data
    data1 = pd.read_csv("gsr_features.csv")
    data2 = pd.read_csv("pupil_features.csv")
    data3 = pd.read_csv("skintemp_features.csv")

    # drop first column as it is identifier
    data1.drop(data1.columns[0], axis=1, inplace=True)
    # drop first two columns as they are repetitive
    data2.drop(data2.columns[0:2], axis=1, inplace=True)
    data3.drop(data3.columns[0:2], axis=1, inplace=True)

    # concatenate three dataframes into one to make it convenient to be dealt with
    data = pd.concat([data1, data2, data3], axis=1)

    # try shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # check the distribution of four levels of depression
    total_targets = data.iloc[:, 0].values
    depression_levels = [0, 1, 2, 3]
    # count frequencies of different levels
    fre_diff_levels = collections.Counter(total_targets)
    heights = []
    for level in depression_levels:
        heights.append(fre_diff_levels[level])

    # plot the x label and the y label
    x_labels = ["None", "Mild", "Moderate", "Severe"]
    plt.xticks(depression_levels, x_labels)
    plt.bar(depression_levels, heights)
    plt.ylabel("Frequency")
    plt.title("Distribution of various depression levels")
    # display the result
    plt.show()

    # randomly split data into training set (70%), validation set (10%) and testing set (20%)
    tmp = np.random.rand(len(data))
    msk_train = tmp < 0.7
    msk_validation = (tmp >= 0.7) & (tmp < 0.8)
    msk_test = tmp >= 0.8
    train_data = data[msk_train]
    validation_data = data[msk_validation]
    test_data = data[msk_test]
    train_data.to_csv("train.csv", sep=',', index=False)
    validation_data.to_csv("validation.csv", sep=',', index=False)
    test_data.to_csv("test.csv", sep=',', index=False)
