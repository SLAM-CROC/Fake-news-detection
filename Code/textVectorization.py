import preprocessing
import visualizeData
from tensorflow.keras.preprocessing.text import Tokenizer


def vectorize(texts):
    for i in texts:
        tokenizer = Tokenizer(filters=' ')  # set num_words to default None, process all words
        tokenizer.fit_on_texts([i[1]])
        i[1] = tokenizer.texts_to_sequences([i[1]])[0]
    for i in texts:
        i[i] = pad_sequences([i[1]], maxlen=1000)



