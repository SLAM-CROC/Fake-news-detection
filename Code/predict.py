import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
import textVectorization
from keras.models import load_model

model = load_model("model_LSTM_text.h5")

features_data, labels_data = preprocessing.clean_data('news.csv')
print("News Tittle: " + str(features_data[0:1][0][0]))
features_data = preprocessing.extract_text(features_data)
features_data, tokenizer = textVectorization.vectorize(features_data)
features_data = pad_sequences(features_data, maxlen=1000)

predict = model.predict(features_data[0:1])
print("Likelihood of REAL: " + str(predict))
print(labels_data[0:1])

