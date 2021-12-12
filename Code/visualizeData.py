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


if __name__ == '__main__':

    # Cleaning data
    # Load dataset from csv file and delete the missing data points
    # Remove url in the text
    # Remove newline signals from text
    # Remove numbers from text
    # Remove punctuations from text
    # Convert all letters into lowercase
    # Tokenization text
    # Remove stopwords from text
    # Normalization: including stemming and Lemmatization
    # Remove words whose length are equal or less than 2
    features_data, labels_data = preprocessing.clean_data('news.csv')

    # Divide the data into two categories according to the label and observe the amount of data in each category
    features_real_data, features_fake_data = split_features(features_data, labels_data)

    # Draw a histogram of frequency distribution and observe dataset
    draw_plot(features_real_data)
    draw_plot(features_fake_data)
