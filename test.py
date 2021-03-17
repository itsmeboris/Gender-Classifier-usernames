import pandas as pd
from cnn import turn_to_vectors, load_vectors
from keras.models import load_model
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from cnn import male_female_split


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


labels = ['male', 'female']
vectors = load_vectors()
max_len = 18
threshold = 200000

test_name = 'anime'
training_names = ['fxp', 'twitter', 'okcupid', 'training']

df = pd.read_csv(f'datasets/{test_name}_users.csv')
df['name'] = df.name.apply(lambda name: ','.join(name.split(' ')))
if test_name is 'anime':
    df = male_female_split(df, 0.6)
if test_name is 'fxp':
    df = male_female_split(df, 0.12)
if len(df) > threshold:
    df, _ = train_test_split(df, train_size=0.8)

test_X, test_Y = turn_to_vectors(df, max_len, vectors)

models = [load_model(f'Models/{name}_lstm_{max_len}.h5') for name in training_names]
for model, name in zip(models, training_names):
    df[name] = ['male' if x == 0 else 'female' for x in np.argmax(model.predict(test_X, verbose=1), axis=1)]

df['ensemble_label'] = df.apply(ensemble_label, axis=1)

y_true = list(df['gender'])
y_pred = list(df['ensemble_label'])
cm = confusion_matrix(y_true, y_pred, labels=labels)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print(f'The accuracy of the classifier train of ensemble is: {accuracy_score(y_true, y_pred)}')

stacked_model = load_model(f'Models/stacked_ensemble_lstm_18.h5')
yhat = predict_stacked_model(stacked_model, test_X)
yhat = np.argmax(yhat, axis=1)
acc = accuracy_score(np.argmax(test_Y, axis=1), yhat)
print('Stacked Test Accuracy: %.3f' % acc)
