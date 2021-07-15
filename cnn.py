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

    def build_model(self, name='default'):
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
        model = get_model(name)
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=metrics)
        # if self.debug:
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
        history = self.model.fit(
            turn_to_vectors_gen(train, max_len, vectors, batch_size),
            steps_per_epoch=len(train) // batch_size,
            batch_size=batch_size,
            epochs=self.epochs,
            validation_data=turn_to_vectors_gen(test, max_len, vectors, batch_size),
            validation_batch_size=batch_size,
            validation_steps=len(test) // batch_size,
            callbacks=[checkpoint(self.train_name, max_len),
                       tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)],
            verbose=self.debug)
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
        return self.model.predict(x, verbose=0)
