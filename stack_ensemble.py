from keras.models import load_model, Model
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.utils import plot_model
import pandas as pd
from cnn import turn_to_vectors, load_vectors, checkpoint, plot_history, male_female_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# load models from file
def load_all_models(models, max_len):
    all_models = []
    for name in models:
        # define filename for this ensemble
        filename = f'Models/{name}_lstm_{max_len}.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        model.input._name = 'ensemble_' + str(i + 1) + '_' + model.input.name
        model.input.type_spec._name = model.input.name
        model.input_names[0] = model.input.name
        model.input._keras_history.layer._name = model.input.name
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    # plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, df, vectors):
    train, test = train_test_split(df, train_size=0.8)
    train_X, train_Y = turn_to_vectors(train, max_len, vectors)
    test_X, test_Y = turn_to_vectors(test, max_len, vectors)
    # prepare input data
    X = [train_X for _ in range(len(model.input))]
    test_X = [test_X for _ in range(len(model.input))]
    # encode output data
    # fit model
    batch_size = len(train) // 100
    history = model.fit(X, train_Y, batch_size=batch_size, epochs=50, verbose=1, callbacks=checkpoint('stacked_ensemble', max_len), validation_data=(test_X, test_Y))
    plot_history(history, 'stacked_ensemble', max_len)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=1)


all_models = ['fxp', 'twitter', 'okcupid', 'training']
max_len = 18

test_name = 'anime'
df = pd.read_csv(f'datasets/{test_name}_users.csv')
if test_name is 'anime':
    df = male_female_split(df, 0.6)
if test_name is 'fxp':
    df = male_female_split(df, 0.12)
_, test = train_test_split(df, train_size=0.8)
print(test['gender'].value_counts())
vectors = load_vectors()
# load all models
members = load_all_models(all_models, max_len)
print('Loaded %d models' % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, test, vectors)
# make predictions and evaluate
test_X, test_Y = turn_to_vectors(test, max_len, vectors)
yhat = predict_stacked_model(stacked_model, test_X)
yhat = np.argmax(yhat, axis=1)
acc = accuracy_score(np.argmax(test_Y, axis=1), yhat)
print('Stacked Test Accuracy: %.3f' % acc)