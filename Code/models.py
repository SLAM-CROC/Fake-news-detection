from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization
import tensorflow as tf
from keras.layers import LSTM


# Define Early Stop to prevent over fitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


def cnn_model(x_train, y_train, x_test, y_test, vocab_size, EMBEDDING_DIM, embedding_vectors, maxlen):
    model = Sequential()

    model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM,
                        weights=[embedding_vectors], input_length=maxlen, trainable=False))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(3, 3, padding='same'))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=6, callbacks=[early_stop])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_acc


def mlp_model(x_train, y_train, x_test, y_test, vocab_size, EMBEDDING_DIM, embedding_vectors, maxlen):
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM,
                        weights=[embedding_vectors], input_length=maxlen, trainable=False))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=6, callbacks=[early_stop])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_acc


def lstm_model(x_train, y_train, x_test, y_test, vocab_size, EMBEDDING_DIM, embedding_vectors, maxlen):
    # Defining Neural Network
    model = Sequential()
    # Non-trainable embedding layer
    model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM,
                        weights=[embedding_vectors], input_length=maxlen,
                        trainable=False))
    # LSTM
    model.add(LSTM(units=128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.2, epochs=6, callbacks=[early_stop])

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_acc


def lr_model(x_train, y_train, x_test, y_test):
    pipe2 = Pipeline([('vectorized', CountVectorizer()), ('tfidf', TfidfTransformer()),
                      ('model', LogisticRegressionCV(cv=5, scoring='accuracy', random_state=0, n_jobs=-1, max_iter=300))])
    model2 = pipe2.fit(x_train, y_train)
    result2 = model2.predict(x_test)
    print("LogisticRegressionCV:")
    print("Accuracy: ", accuracy_score(y_test, result2))
    print("F1 Score: ", f1_score(y_test, result2, average='micro'))
    print("\n")


def nb_model(x_train, y_train, x_test, y_test):
    pipe1 = Pipeline([('vectorized', CountVectorizer()), ('tfidf', TfidfTransformer()),
                      ('model', MultinomialNB())])
    model1 = pipe1.fit(x_train, y_train)
    result1 = model1.predict(x_test)
    print("Naive Bayes:")
    print("Accuracy: ", accuracy_score(y_test, result1))
    print("F1 Score: ", f1_score(y_test, result1, average='micro'))
    print("\n")


def knn_model(x_train, y_train, x_test, y_test):
    pipe3 = Pipeline([('vectorized', CountVectorizer()), ('tfidf', TfidfTransformer()),
                      ('model', KNeighborsClassifier(n_neighbors=5))])
    model3 = pipe3.fit(x_train, y_train)
    result3 = model3.predict(x_test)
    print("KNN:")
    print("Accuracy: ", accuracy_score(y_test, result3))
    print("F1 Score: ", f1_score(y_test, result3, average='micro'))
    print("\n")


def svm_model(x_train, y_train, x_test, y_test):
    pipe4 = Pipeline([('vectorized', CountVectorizer()), ('tfidf', TfidfTransformer()),
                      ('model', SVC(kernel='linear', random_state=1))])
    model4 = pipe4.fit(x_train, y_train)
    result4 = model4.predict(x_test)
    print("SVM:")
    print("Accuracy: ", accuracy_score(y_test, result4))
    print("F1 Score: ", f1_score(y_test, result4, average='micro'))
    print("\n")


def decision_tree_model(x_train, y_train, x_test, y_test):
    pipe5 = Pipeline([('vectorized', CountVectorizer()), ('tfidf', TfidfTransformer()),
                      ('model', DecisionTreeClassifier())])
    model5 = pipe5.fit(x_train, y_train)
    result5 = model5.predict(x_test)
    print("Decision Tree:")
    print("Accuracy: ", accuracy_score(y_test, result5))
    print("F1 Score: ", f1_score(y_test, result5, average='micro'))
    print("\n")
