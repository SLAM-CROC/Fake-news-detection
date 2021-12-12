# Deep learning project: Fake new detection
import preprocessing
import models


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

    # Restore data to string
    features_data = preprocessing.recover_to_string(features_data)

    # Drop News Tittle
    features_data = preprocessing.extract_text(features_data)

    # Encode Labels, encode FAKE and Real to 0 and 1
    labels_data = preprocessing.encode_labels(labels_data)

    # Split the data by the proportion 4:1
    (x_train, y_train), (x_test, y_test) = preprocessing.split_data(features_data, labels_data, 0.2)
    y_test = y_test.astype('int')
    y_train = y_train.astype('int')

    # Training Machine Learning Model
    models.lr_model(x_train, y_train, x_test, y_test)
    models.nb_model(x_train, y_train, x_test, y_test)
    models.knn_model(x_train, y_train, x_test, y_test)
    models.svm_model(x_train, y_train, x_test, y_test)
    models.decision_tree_model(x_train, y_train, x_test, y_test)
