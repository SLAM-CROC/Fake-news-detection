import preprocessing
import visualizeData
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def vectorize(texts):
    tokenizer = Tokenizer(filters=' ')  # set num_words to default None, process all words
    for i in texts:
        tokenizer.fit_on_texts([i[1]])       # Note: the parameter should be a list
        i[1] = tokenizer.texts_to_sequences([i[1]])[0]     # Note: the parameter should be a list
    for i in texts:
        i[i] = pad_sequences([i[1]], maxlen=1000)


def find_max_sequence_len(features):
    m = 0
    for i in features:
        if len(i[1]) > m:
            m = len(i[1])
        else:
            continue
    return m

