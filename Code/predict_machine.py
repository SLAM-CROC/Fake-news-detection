import sys
import preprocessing
import textVectorization
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model_LSTM_text.h5")

raw_news = input("Please input a piece of news:")

feature = ['agwegw', raw_news]
preprocessing.remove_url(feature)
preprocessing.remove_newline(feature)
preprocessing.remove_number(feature)
preprocessing.remove_punctuation(feature)
preprocessing.convert_into_lowercase(feature)
preprocessing.tokenization(feature)
preprocessing.remove_stopwords(feature)
preprocessing.normalization(feature)
preprocessing.remove_short_words(feature)
feature, tokenizer = textVectorization.vectorize(feature)
feature = pad_sequences(feature, maxlen=1000)

predict = model.predict(feature)
print(float(predict))

