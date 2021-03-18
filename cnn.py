from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout
from keras.models import Sequential

from utility import *


class VectorCnn:
    def __init__(self, train_name, epochs, debug=False):
        """
        Initialize the basic variables for the CNN
        :param train_name: Data set to train on
        :param epochs: number of epochs to train on
        :param debug: print debug information
        """
        self.train_name = train_name
        self.epochs = epochs
        self.model = None
        self.history = None
        self.debug = debug

    def build_model(self):
        """
        In case you want a good number of hidden layers use this
        formula to calculate it.
        n_s = len(df)
        n_i = vector_size
        n_o = max_len
        alpha = 2
        hidden_layer_size = n_s // (alpha * (n_i + n_o))
        """
        if self.debug:
            print('Build model...')
        model = Sequential()
        model.add(LSTM(512, return_sequences=False, input_shape=(max_len, vector_size)))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if self.debug:
            model.summary()
        self.model = model

    def fit_model(self, train, test):
        """
        Train the CNN model
        :param train: train data set
        :param test: test data set
        """
        if self.model is None:
            print('Please wait')
            self.build_model()
            print('Done building the model, starting to fit')
        if self.debug:
            print(train.gender.value_counts())
            print(test.gender.value_counts())
        train_X, train_Y = turn_to_vectors(train, max_len, vectors)
        test_X, test_Y = turn_to_vectors(test, max_len, vectors)
        batch_size = len(train) // 100
        history = self.model.fit(train_X, train_Y, batch_size=batch_size, epochs=self.epochs,
                                 validation_data=(test_X, test_Y),
                                 callbacks=checkpoint(self.train_name, max_len), verbose=0)
        self.history = history

    def show_history(self, name=None):
        """
        Plot the training history of the model
        """
        if name is None:
            plot_history(self.history, self.train_name)
        else:
            plot_history(self.history, name)

    def predict_stacked_model(self, x):
        """
        :param x: Data to predict
        :return: predicted values
        """
        return self.model.predict(x, verbose=1)
