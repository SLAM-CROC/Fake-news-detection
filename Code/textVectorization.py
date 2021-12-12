from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


def vectorize(extracted):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(extracted)
    extracted = tokenizer.texts_to_sequences(extracted)
    return extracted, tokenizer


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab, EMBEDDING_DIM):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix

