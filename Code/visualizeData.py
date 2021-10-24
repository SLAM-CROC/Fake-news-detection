import preprocessing
import matplotlib.pyplot as plt


def show_dataset(features, labels):
    for i in range(len(features)):
        print((i+1), features[i], labels[i])


def split_features(features, labels):
    features_real = []
    features_fake = []
    for i in range(len(features)):
        if labels[i] == "REAL":
            features_real.append(features[i][1])
        else:
            features_fake.append(features[i][1])
    print("Real data number:" + str(len(features_real)) + "\nFake data number:" + str(len(features_fake)))
    return features_real, features_fake


def draw_plot(data):
    temp = []
    for i in data:
        temp = temp + i
    dict_word = {}
    for i in temp:
        dict_word[i] = dict_word.get(i, 0) + 1
    dict_word = sorted(dict_word.items(), key=lambda x: x[1], reverse=True)[:11]
    for i in dict_word:
        plt.bar((i[0],), (i[-1],))
    plt.show()


