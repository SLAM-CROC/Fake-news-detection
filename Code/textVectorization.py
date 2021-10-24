import preprocessing
import visualizeData
from tensorflow.keras.preprocessing.text import Tokenizer

# features_data, labels_data = preprocessing.load_csv('news.csv')
# preprocessing.processing(features_data)
# preprocessing.recover_to_string(features_data)


def vectorize(texts):
    for i in texts:
        tokenizer = Tokenizer(filters=' ')  # set num_words to default None, process all words
        tokenizer.fit_on_texts([i[1]])
        i[1] = tokenizer.texts_to_sequences([i[1]])[0]


# vectorize(features_data)
# visualizeData.show_dataset(features_data, labels_data)
