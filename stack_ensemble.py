from utility import *


class StackedEnsemble:
    def __init__(self, all_models, train_name, epochs, debug=False):
        """
        Initialize the basic variables for the stacked ensemble
        :param all_models: an array containing the names of all the models to load
        :param train_name: the data set to train on
        :param epochs: number of epochs to train on
        :param debug: print debug information
        """
        self.members = load_all_models(all_models)
        self.len_members = len(self.members)
        self.epochs = epochs
        self.debug = debug
        print('Loaded %d models' % len(self.members))
        self.train_name = train_name
        self.model = None
        self.history = None

    # define stacked model from multiple member input models
    def build_model(self):
        # update all layers in all models to not be trainable
        if self.debug:
            print('Build model...')
            print('Starting to change layer names to unique names')
        for i in range(self.len_members):
            model = self.members[i]
            model.input._name = 'ensemble_' + str(i + 1) + '_' + model.input.name
            model.input.type_spec._name = model.input.name
            model.input_names[0] = model.input.name
            model.input._keras_history.layer._name = model.input.name
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unique layer name' issue
                layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
            self.members[i] = model
        if self.debug:
            print('Done changing names')
        # define multi-headed input
        ensemble_visible = [model.input for model in self.members]
        # concatenate merge output from each model
        ensemble_outputs = [model.output for model in self.members]
        merge = concatenate(ensemble_outputs)
        hidden_1 = Dense(10, activation='relu')(merge)
        hidden_2 = Dense(10, activation='relu')(hidden_1)
        output = Dense(2, activation='softmax')(hidden_2)
        model = Model(inputs=ensemble_visible, outputs=output)
        # plot graph of ensemble
        # compile
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=metrics)
        # if self.debug:
        model.summary()
        # plot_model(model, show_shapes=True, to_file=f'stacked_model_graph_{self.train_name}.png')
        self.model = model

    # fit a stacked model
    def fit_model(self, train, test):
        """
        Train the stacked ensemble model
        :param train: train data set
        :param test: test data set
        """
        if self.model is None:
            print('Please wait')
            self.build_model()
            print('Done building the model, starting to fit')
        history = self.model.fit(
            turn_to_vectors_gen(train, max_len, vectors, batch_size, copies=len(self.model.input)),
            steps_per_epoch=len(train) // batch_size,
            batch_size=batch_size,
            epochs=self.epochs,
            validation_data=turn_to_vectors_gen(test, max_len, vectors, batch_size, copies=len(self.model.input)),
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

    # make a prediction with a stacked model
    def predict_stacked_model(self, input_x):
        """
        :param input_x: Data to predict
        :return: predicted values
        """
        # prepare input data
        X = [input_x for _ in range(len(self.model.input))]
        # make prediction
        return self.model.predict(X, verbose=0)
