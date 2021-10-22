import re
import tensorflow as tf
import csv
import numpy as np


# Load csv file and remove blank data
def load_csv(filename):
    features = []
    labels = []
    count = 1
    blank_data_number_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            # print(line[1:3])
            if line[1] == 'title':
                count += 1
                continue
            elif line[1] == ' ' or line[2] == ' ':
                blank_data_number_list.append(count)
                count += 1
                continue
            features.append(line[1:3])
            labels.append(line[3])
            count += 1
    print(blank_data_number_list)
    return features, labels


def split_data(features, labels, test_proportion):
    index = int(len(features) * (1 - test_proportion))
    train_x, train_y = np.array(features[:index]), np.array(labels[:index])
    test_x, test_y = np.array(features[index:]), np.array(labels[index:])
    return (train_x, train_y), (test_x, test_y)


def remove_url(features):
    count = 0
    for i in features:
        count += 1
        if 'http://' in i[1] or 'https://' in i[1]:
            i[1] = re.sub(r'http\S+', ' ', i[1])
            print(count)
            print(i)


features_data, labels_data = load_csv('news.csv')

# print(len(features_data))
# print(features_data[0][0])

remove_url(features_data)


# (x_train, y_train), (x_test, y_test) = split_data(features_data, labels_data, 0.2)
