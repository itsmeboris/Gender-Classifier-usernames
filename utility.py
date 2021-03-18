import operator
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

plt.style.use('ggplot')


def male_female_split(df, factor):
    female = df[df['gender'] == 'female']
    male = df[df['gender'] == 'male']
    msk = np.random.rand(len(male)) < factor
    male, rest = male[msk], male[~msk]
    return pd.concat([female, male])[['name', 'gender']]


def load_split_data(name, debug=False):
    df = pd.read_csv(f'datasets/{name}_users.csv')
    if debug:
        print(df.groupby('gender')['name'].count())
    df['name'] = df.name.apply(lambda name: ','.join(name.split(' ')))
    if name is 'anime':
        df = male_female_split(df, 0.6)
    if name is 'fxp':
        df = male_female_split(df, 0.12)
    if len(df) > threshold:
        df, _ = train_test_split(df, train_size=0.8)
    return df


def plot_history(history, name):
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


def plot_data(val_acc, train_name):
    index = list(range(1, len(val_acc) + 1))
    plt.figure(figsize=(12, 5))
    plt.plot(index, val_acc, 'b', label='Training acc')
    plt.title(f'10-Fold accuracy scores')
    plt.legend()
    plt.savefig(f'images/{train_name}_10_fold_accuracy_score', bbox_inches="tight", transparent=True)
    plt.show()


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
    return tf.keras.callbacks.ModelCheckpoint(f'Models/temp_models/{name}_lstm_{max_len}.h5',
                                              monitor='val_accuracy', verbose=0,
                                              save_best_only=True, mode='max')


# load models from file
def load_all_models(models):
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


def ensemble_label(row):
    d = {'male': 0, 'female': 0}
    for name in training_names:
        d[row[name]] += 1
    return max(d.items(), key=operator.itemgetter(1))[0]


def predict_stacked_model(model, input_x):
    # prepare input data
    X = [input_x for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=1)


def get_labels(predicted):
    return [labels[x] for x in np.argmax(predicted, axis=1)]


def plot_test(df, y_true_label, y_predicted_label):
    y_true = list(df[y_true_label])
    y_pred = list(df[y_predicted_label])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(f'Confusion matrix of the classifier {y_predicted_label}')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print(f'The accuracy of the classifier {y_predicted_label} is: {accuracy_score(y_true, y_pred)}')


def save_best_model(models, train_name):
    val_acc = [max(model.history.history['val_accuracy']) for _, model in models.items()]
    plot_data(val_acc, train_name)
    highest_accuracy = max(val_acc)
    index = val_acc.index(highest_accuracy)
    best_model = models[index]
    best_model.model.save(f'Models/{train_name}_lstm_{max_len}.h5')
    best_model.show_history(train_name)


labels = ['male', 'female']
all_sets = ['fxp', 'twitter', 'okcupid', 'training', 'anime']
training_names = ['fxp', 'twitter', 'okcupid', 'training', 'anime']
max_len = 18
threshold = 200000
vectors = load_vectors()
vector_size = len(vectors['0'][0])
