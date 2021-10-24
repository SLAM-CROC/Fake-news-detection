import re
import csv
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


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
    print("Loaded csv file, and there are " + str(len(blank_data_number_list)) + " blank text have been removed "
                                                                                 "from dataset\n" + str(len(features)) +
          " data points in total")
    return features, labels


def split_data(features, labels, test_proportion):
    index = int(len(features) * (1 - test_proportion))
    train_x, train_y = np.array(features[:index]), np.array(labels[:index])
    test_x, test_y = np.array(features[index:]), np.array(labels[index:])
    return (train_x, train_y), (test_x, test_y)


def remove_url(features):
    count = 0
    for i in features:
        if 'http://' in i[1] or 'https://' in i[1]:
            count += 1
            i[1] = re.sub(r'http\S+', ' ', i[1])
    print("There are "+str(count)+" url have been removed from text")


def remove_newline(features):
    for i in features:
        i[1] = i[1].replace('\n', ' ').replace('\r', ' ').replace('\n\n', ' ')
    print("Newline symbols have been removed from text")


def remove_number(features):
    for i in features:
        i[1] = re.sub(r'\d+', ' ', i[1])
    print("Numbers have been removed from text")


def remove_punctuation(features):
    for i in features:
        i[1] = re.sub('[^a-zA-Z]', ' ', i[1])
    print("Punctuations have been removed from text")


def convert_into_lowercase(features):
    for i in features:
        i[1] = i[1].lower()
    print("All text have been converted into lowercase")


def tokenization(features):
    for i in features:
        i[1] = word_tokenize(i[1])
    print("Preformed tokenization")


def remove_stopwords(features):
    stop_words = set(stopwords.words('english'))
    for i in features:
        i[1] = [words for words in i[1] if not words in stop_words]
    print("Stopwords have been removed")


def normalization(features):
    stemmer = PorterStemmer()
    lemma = WordNetLemmatizer()
    for i in features:
        i[1] = [stemmer.stem(word) for word in i[1]]
    for i in features:
        i[1] = [lemma.lemmatize(word=word, pos='v') for word in i[1]]
    print("Text has been normalized")


def remove_short_words(features):
    for i in features:
        i[1] = [word for word in i[1] if len(word) > 2]
    print("Short words have been removed")


def recover_to_string(features):
    for i in features:
        i[1] = ' '.join(i[1])
    print('The text have been recovered from words to string')


def processing(features):
    remove_url(features)
    remove_newline(features)
    remove_number(features)
    remove_punctuation(features)
    convert_into_lowercase(features)
    tokenization(features)
    remove_stopwords(features)
    normalization(features)
    remove_short_words(features)
