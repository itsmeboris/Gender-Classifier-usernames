import operator
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Dense, LSTM, GlobalMaxPool1D, GlobalAveragePooling1D, Input, concatenate, Attention, Dropout, \
    Conv1D, Bidirectional, GRU, Activation, Add, MaxPooling1D, Flatten, AveragePooling1D
from keras.models import Sequential, Model

plt.style.use('ggplot')


def male_female_split(df, factor):
    female = df[df['gender'] == 'female']
    male = df[df['gender'] == 'male']
    msk = np.random.rand(len(male)) < factor
    male, rest = male[msk], male[~msk]
    return pd.concat([female, male])[['name', 'gender']]


def load_split_data(name, debug=False, split=False):
    df = pd.read_csv(f'datasets/{name}_users.csv')
    if debug:
        print(df.groupby('gender')['name'].count())
    df['name'] = df.name.apply(lambda name: ','.join(name.split(' ')))
    if split:
        if name is 'anime':
            df = male_female_split(df, 0.6)
        if name is 'fxp':
            df = male_female_split(df, 0.12)
    # if len(df) > threshold:
    #     df, _ = train_test_split(df, train_size=0.3)
    return df


def plot_history(history, name, use_metrics=None):
    if use_metrics is None:
        use_metrics = metrics
    for metric in use_metrics:
        value = history.history[metric]
        val_value = history.history[f'val_{metric}']
        x = range(1, len(value) + 1)
        plt.figure(figsize=(12, 5))
        plt.plot(x, value, label=f'Training {metric}')
        plt.plot(x, val_value, label=f'Validation {metric}')
        plt.title(f'Training and validation {metric} on {name}')
        plt.legend()
        plt.savefig(f'images/{name}_lstm_{max_len}_train_{metric}', bbox_inches="tight", transparent=False)
        plt.close()
    print(f"Accuracy score of {max(history.history['val_accuracy'])} was achieved")


def plot_data(val_acc, train_name):
    index = list(range(1, len(val_acc) + 1))
    plt.figure(figsize=(12, 5))
    plt.plot(index, val_acc, 'b', label='Training acc')
    plt.title(f'10-Fold accuracy scores')
    plt.legend()
    plt.savefig(f'images/{train_name}_10_fold_accuracy_score', bbox_inches="tight", transparent=True)
    plt.close()


def turn_to_vectors(df, max_len, vectors, debug=False):
    X, Y = [], []
    trunc_train_name = [str(i)[0:max_len] for i in df.name]
    for i in trunc_train_name:
        tmp = [vectors[j][0] for j in str(i)]
        for k in range(0, max_len - len(str(i))):
            tmp.append(vectors['%'][0])
        X.append(tmp)
    for i in df.gender:
        if i == 'male':
            Y.append([1, 0])
        else:
            Y.append([0, 1])
    if debug:
        print(np.asarray(X).shape)
        print(np.asarray(Y).shape)
    return np.asarray(X), np.asarray(Y)


def turn_to_vectors_gen(df, max_len, vectors, batch_size, debug=False, copies=None):
    while True:
        for batch in range(0, len(df), batch_size):
            X, Y = [], []
            trunc_train_name = [str(i)[0:max_len] for i in df.iloc[batch: batch+batch_size].name]
            for i in trunc_train_name:
                tmp = [vectors[j][0] for j in str(i)]
                for k in range(0, max_len - len(str(i))):
                    tmp.append(vectors['%'][0])
                X.append(tmp)
            for i in df.iloc[batch: batch+batch_size].gender:
                if i == 'male':
                    Y.append([1, 0])
                else:
                    Y.append([0, 1])
            if debug:
                print(np.asarray(X).shape)
                print(np.asarray(Y).shape)
            yield np.asarray(X) if copies is None else [np.asarray(X) for _ in range(copies)], np.asarray(Y)


def load_vectors():
    vectors = {}
    with open('glove.840B.300d-char.txt', 'r') as f:
        for line in f:
            line_split = line.split(" ")
            vec = np.array(line_split[1:], dtype=float)
            word = line_split[0]

            for char in word:
                if ord(char) < 128:
                    if char in vectors:
                        vectors[char] = (vectors[char][0] + vec,
                                         vectors[char][1] + 1)
                    else:
                        vectors[char] = (vec, 1)
    return vectors


def checkpoint(name, max_len):
    return tf.keras.callbacks.ModelCheckpoint(f'Models/temp_models/{name}_lstm_{max_len}.h5',
                                              monitor='val_accuracy', verbose=0,
                                              save_best_only=True, mode='max')


# load models from file
def load_all_models(models):
    all_models = []
    for name in models:
        # define filename for this ensemble
        filename = f'Models/{name}_lstm_{max_len}.h5'
        try:
            # load model from file
            model = load_model(filename)
            # add to list of members
            all_models.append(model)
        except OSError as oe:
            pass
        # print('>loaded %s' % filename)
    return all_models


def ensemble_label(row, prefix=''):
    d = {'male': 0, 'female': 0}
    for name in training_names:
        try:
            d[row[prefix + name]] += 1
        except:
            pass
    return max(d.items(), key=operator.itemgetter(1))[0]


def predict_stacked_model(model, input_x):
    # prepare input data
    X = [input_x for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


def get_labels(predicted):
    return [labels[x] for x in np.argmax(predicted, axis=1)]


def plot_test(df, y_true_label, y_predicted_label):
    y_true = list(df[y_true_label])
    y_pred = list(df[y_predicted_label])
    accuracy = accuracy_score(y_true, y_pred)
    print(f'The accuracy of the classifier {y_predicted_label} is: {accuracy}')
    return accuracy


def save_best_model(models, train_name):
    val_acc = [max(model.history.history['val_accuracy']) for model in models]
    plot_data(val_acc, train_name)
    highest_accuracy = max(val_acc)
    index = val_acc.index(highest_accuracy)
    best_model = models[index]
    best_model.model.save(f'Models/{train_name}_lstm_{max_len}.h5')
    best_model.show_history(train_name)


def get_DPCNN(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    # DPCNN
    input_layer = Input(shape=(max_len, vector_size), )
    # X = Embedding(max_features, embed_size, weights=[embedding_matrix],
    #              trainable=False)(input_layer)
    # first block
    X_shortcut1 = input_layer
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X_shortcut1)
    X = Activation('relu')(X)
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X)
    X = Activation('relu')(X)

    # connect shortcut to the main path
    X = Activation('relu')(X_shortcut1)  # pre activation
    X = Add()([X_shortcut1, X])
    X = MaxPooling1D(pool_size=3, strides=2, padding='valid')(X)

    # second block
    X_shortcut2 = X
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X)
    X = Activation('relu')(X)
    X = Conv1D(filters=recurrent_units, kernel_size=2, strides=3)(X)
    X = Activation('relu')(X)

    # connect shortcut to the main path
    X = Activation('relu')(X_shortcut2)  # pre activation
    X = Add()([X_shortcut2, X])
    X = MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)

    # Output
    X = Flatten()(X)
    X = Dense(nb_classes, activation='sigmoid')(X)

    model = Model(inputs=input_layer, outputs=X, name='dpcnn')
    return model


def get_gru_best(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    # gru_best
    input_layer = Input(shape=(max_len, vector_size), )
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                          recurrent_dropout=dropout_rate))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                          recurrent_dropout=dropout_rate))(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a, x_b], axis=1)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_LSTM_CONV(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    # LSTM + CONV
    input_layer = Input(shape=(max_len, vector_size))
    x = LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
             recurrent_dropout=dropout_rate)(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = Conv1D(filters=300,
               kernel_size=5,
               padding='valid',
               activation='tanh',
               strides=1)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a, x_b])
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


def get_bidirectional_LSTM(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    # bidirectional_LSTM
    input_layer = Input(shape=(max_len, vector_size), )
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=dropout_rate))(input_layer)
    x = Dropout(dropout_rate)(x)
    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a, x_b])
    x = Dense(dense_size, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


def get_bid_GRU_bid_LSTM(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes):
    # _bid_GRU_bid_LSTM
    input_layer = Input(shape=(max_len, vector_size), )
    x = Bidirectional(GRU(recurrent_units, return_sequences=True, dropout=dropout_rate,
                          recurrent_dropout=recurrent_dropout_rate))(input_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=dropout_rate,
                           recurrent_dropout=recurrent_dropout_rate))(x)

    x_a = GlobalMaxPool1D()(x)
    x_b = GlobalAveragePooling1D()(x)
    x = concatenate([x_a, x_b])

    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(nb_classes, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def get_model(name, recurrent_units=512, dropout_rate=0.3, recurrent_dropout_rate=0.3, dense_size=300, nb_classes=2):
    if name == 'DPCNN':
        return get_DPCNN(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    if name == 'gru_best':
        return get_gru_best(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    if name == 'LSTM_CONV':
        return get_LSTM_CONV(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    if name == 'bidirectional_LSTM':
        return get_bidirectional_LSTM(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    if name == 'bid_GRU_bid_LSTM':
        return get_bid_GRU_bid_LSTM(recurrent_units, dropout_rate, recurrent_dropout_rate, dense_size, nb_classes)
    # Default
    model = Sequential()
    model.add(LSTM(recurrent_units, activation='sigmoid', return_sequences=True, input_shape=(max_len, vector_size)))
    model.add(AveragePooling1D())
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    return model


def vec_generator(df, max_len, vectors, batch_size=64, copies=None):
    while True:
        for batch in range(0, len(df), batch_size):
            X = []
            trunc_train_name = [str(i)[0:max_len] for i in df.iloc[batch: batch + batch_size].name]
            for i in trunc_train_name:
                tmp = [vectors[j][0] for j in str(i)]
                for k in range(0, max_len - len(str(i))):
                    tmp.append(vectors['%'][0])
                X.append(tmp)
            yield np.asarray(X) if copies is None else [np.asarray(X) for _ in range(copies)]


labels = ['male', 'female']
all_sets = ['fxp', 'twitter', 'okcupid', 'anime', 'entity']
training_names = ['fxp', 'twitter', 'okcupid', 'anime', 'entity']
max_len = 18
threshold = 200000
batch_size = 64
vectors = load_vectors()
vector_size = len(vectors['0'][0])
VOCAB_SIZE = len(vectors)
metrics = ['accuracy', 'loss']
model_types = ['default', 'LSTM_CONV', 'gru_best', 'bid_GRU_bid_LSTM', 'bidirectional_LSTM']
