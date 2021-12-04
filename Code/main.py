# Deep learning project: Fake new detection
import preprocessing
import visualizeData
import textVectorization
import models


if __name__ == '__main__':
    # Preprocessing data
    # Load dataset from csv file and delete the missing data points
    features_data, labels_data = preprocessing.load_csv('news.csv')

    # Remove url in the text
    preprocessing.remove_url(features_data)

    # Remove newline signals from text
    preprocessing.remove_newline(features_data)

    # Remove numbers from text
    preprocessing.remove_number(features_data)

    # Remove punctuations from text
    preprocessing.remove_punctuation(features_data)

    # Convert all letters into lowercase
    preprocessing.convert_into_lowercase(features_data)

    # Tokenization text
    preprocessing.tokenization(features_data)

    # Remove stopwords from text
    preprocessing.remove_stopwords(features_data)

    # Normalization: including stemming and Lemmatization
    preprocessing.normalization(features_data)

    # Remove words whose length are equal or less than 2
    preprocessing.remove_short_words(features_data)

    # Divide the data into two categories according to the label and observe the amount of data in each category
    features_real_data, features_fake_data = visualizeData.split_features(features_data, labels_data)

    # Draw a histogram of frequency distribution and observe dataset
    visualizeData.draw_plot(features_real_data)
    visualizeData.draw_plot(features_fake_data)

    # Restore data to string
    preprocessing.recover_to_string(features_data)

    # Preform text vectorization
    textVectorization.vectorize(features_data)

    # encode FAKE and Real to 0 and 1
    labels_data = preprocessing.encode_labels(labels_data)

    # Output the dataset
    # visualizeData.show_dataset(features_data, labels_data)

    # Split the data by the proportion 4:1
    (x_train, y_train), (x_test, y_test) = preprocessing.split_data(features_data, labels_data, 0.2)

    x_train = preprocessing.extract_text(x_train)
    x_test = preprocessing.extract_text(x_test)

    models.mlp_model(x_train, y_train)

    models.cnn_model(x_train, y_train)

    models.lstm_model(x_train, y_train)

    models.lr_model(x_train, y_train, x_test, y_test)

    models.nb_model(x_train, y_train, x_test, y_test)

    models.knn_model(x_train, y_train, x_test, y_test)

    models.svm_model(x_train, y_train, x_test, y_test)

    models.decision_tree_model(x_train, y_train, x_test, y_test)
