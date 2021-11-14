from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization
from keras.layers import LSTM
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def cnn_model(x_train, y_train):
    model = Sequential()
    model.add(Embedding(output_dim=32, input_dim=2000, input_length=1000))
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


def mlp_model(x_train, y_train):
    model = Sequential()
    model.add(Embedding(output_dim=32, input_dim=2000, input_length=1000))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation="sigmoid"))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2)


def lstm_model(x_train, y_train):
    model = Sequential()
    model.add(Embedding(output_dim=32, input_dim=2000, input_length=1000))
    model.add(LSTM(units=128, return_sequences=True, input_shape=(7, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=64))
    model.add(Dense(units=2, activation="sigmoid"))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2)


def lr_model(x_train, y_train, x_test, y_test):
    lr = Pipeline([('tfidf', TfidfTransformer()),
                   ('model',
                    LogisticRegressionCV(cv=5, scoring='accuracy', random_state=0, n_jobs=-1, verbose=3, max_iter=300)),
                   ])
    lr.fit(x_train, y_train)

    y_pred_lr = lr.predict(x_test)

    print('accuracy %s' % accuracy_score(y_pred_lr, y_test))


def nb_model(x_train, y_train, x_test, y_test):
    nb = Pipeline([('tfidf', TfidfTransformer()),
                   ('model', MultinomialNB()),
                   ])
    nb.fit(x_train, y_train)

    y_pred_nb = nb.predict(x_test)

    print('accuracy %s' % accuracy_score(y_pred_nb, y_test))


def knn_model(x_train, y_train, x_test, y_test):
    knn = Pipeline([('tfidf', TfidfTransformer()),
                    ('model', KNeighborsClassifier(n_neighbors=5)),
                    ])
    knn.fit(x_train, y_train)

    y_pred_knn = knn.predict(x_test)

    print('accuracy %s' % accuracy_score(y_pred_knn, y_test))


def svm_model(x_train, y_train, x_test, y_test):
    svm = Pipeline([('tfidf', TfidfTransformer()),
                    ('model', SVC(kernel='linear', random_state=1)),
                    ])
    svm.fit(x_train, y_train)

    y_pred_svm = svm.predict(x_test)

    print('accuracy %s' % accuracy_score(y_pred_svm, y_test))


def decision_tree_model(x_train, y_train, x_test, y_test):
    dt = Pipeline([('tfidf', TfidfTransformer()),
                   ('model', DecisionTreeClassifier()),
                   ])
    dt.fit(x_train, y_train)

    y_pred_dt = dt.predict(x_test)

    print('accuracy %s' % accuracy_score(y_pred_dt, y_test))


