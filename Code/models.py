from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization


def cnn_model(x_train, y_train):
    model = Sequential()
    model.add(Embedding(output_dim=32, input_dim=2000, input_length=50))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2)




