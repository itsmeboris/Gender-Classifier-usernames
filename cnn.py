import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf

plt.style.use('ggplot')


def plot_history(history, name, max_len):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title(f'Training and validation accuracy on {name}')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title(f'Training and validation loss on {name}')
    plt.legend()
    plt.savefig(f'images/{name}_lstm_{max_len}_train', bbox_inches="tight", transparent=True)
    plt.show()
    print(f'Accuracy score of {max(val_acc)} was achieved')


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
    return tf.keras.callbacks.ModelCheckpoint(f'Models/{name}_lstm_{max_len}.h5',
                                              monitor='val_accuracy', verbose=1,
                                              save_best_only=True, mode='max')


def train(debug=False, vectors=None, max_len=30, epochs=50):
    if vectors is None:
        vectors = load_vectors()
    vector_size = len(vectors['0'][0])
    training_names = ['fxp', 'twitter', 'training', 'okcupid']
    for name in training_names:
        df = pd.read_csv(f'datasets/{name}_users.csv')
        if debug:
            print(df.groupby('gender')['name'].count())
        df['name'] = df.name.apply(lambda name: ','.join(name.split(' ')))
        female = df[df['gender'] == 'female']
        male = df[df['gender'] == 'male']

        if name is 'fxp':
            msk = np.random.rand(len(male)) < 0.12
            male, rest = male[msk], male[~msk]

        df = pd.concat([female, male])[['name', 'gender']]
        train, test = train_test_split(df, train_size=0.8)
        if debug:
            print(train.gender.value_counts())
            print(test.gender.value_counts())

        train_X, train_Y = turn_to_vectors(train, max_len, vectors)
        test_X, test_Y = turn_to_vectors(test, max_len, vectors)

        n_s = len(df)
        n_i = vector_size
        n_o = max_len
        alpha = 2
        hidden_layer_size = n_s // (alpha * (n_i + n_o))
        print(f'Number of hidden layers is: {hidden_layer_size}')

        print('Build model...')
        model = Sequential()
        model.add(LSTM(512, return_sequences=False, input_shape=(max_len, vector_size)))
        model.add(Dropout(0.2))
        # model.add(LSTM(512, return_sequences=False))
        # model.add(Dropout(0.2))
        # model.add(Dense(hidden_layer_size, activation='relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if debug:
            model.summary()

        batch_size = len(train) // 100
        history = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_Y),
                            callbacks=checkpoint(name, max_len), verbose=0)
        plot_history(history, name, max_len)


train(debug=False, vectors=load_vectors(), max_len=18, epochs=100)
