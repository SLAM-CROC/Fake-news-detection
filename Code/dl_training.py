# Deep learning project: Fake new detection
import preprocessing
import textVectorization
import models
import gensim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold


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

    # Drop News Tittle
    features_data = preprocessing.extract_text(features_data)

    # Dimension of vectors we are generating
    EMBEDDING_DIM = 100

    # Creating Word Vectors by Word2Vec Method
    w2v_model = gensim.models.Word2Vec(sentences=features_data, size=EMBEDDING_DIM, window=5, min_count=1)

    # Using keras’ built-in Tokenizer to use Unique Numbers to Vectorize
    # Represent each word by its index in the words dictionary.
    features_data, tokenizer = textVectorization.vectorize(features_data)

    # Create weight matrix
    embedding_vectors = textVectorization.get_weight_matrix(w2v_model, tokenizer.word_index, EMBEDDING_DIM)

    # After analysing, we choose 1000 for text padding to ensure that
    # the data contains most of the information at the same time saving training time.
    max_len = 1000
    features_data = pad_sequences(features_data, maxlen=max_len)

    # Encode Labels, encode FAKE and Real to 0 and 1
    labels_data = preprocessing.encode_labels(labels_data)

    # Split the data by the proportion 4:1
    (x_train, y_train), (x_test, y_test) = preprocessing.split_data(features_data, labels_data, 0.1)

    # Vocabulary Size
    vocab_size = len(tokenizer.word_index) + 1

    # Training Deep Learning Model
    test_acc = models.mlp_model(x_train, y_train, x_test, y_test, vocab_size, EMBEDDING_DIM, embedding_vectors, max_len)
    print('\nTest accuracy:', test_acc)

    test_acc = models.cnn_model(x_train, y_train, x_test, y_test, vocab_size, EMBEDDING_DIM, embedding_vectors, max_len)
    print('\nTest accuracy:', test_acc)

    test_acc = models.cnn_model(x_train, y_train, x_test, y_test, vocab_size, EMBEDDING_DIM, embedding_vectors, max_len)
    print('\nTest accuracy:', test_acc)

    # Cross Validation
    kf = KFold(n_splits=3)
    L = []
    for train_index, test_index in kf.split(features_data, labels_data):
        x_train, x_test, y_train, y_test = features_data[train_index], features_data[test_index], \
                                           labels_data[train_index], labels_data[test_index]
        test_acc = models.cnn_model(x_train, y_train, x_test, y_test, vocab_size,
                                    EMBEDDING_DIM, embedding_vectors, max_len)
        L.append(test_acc)
    print(L)
