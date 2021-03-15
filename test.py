import pandas as pd
from cnn import turn_to_vectors, load_vectors
from keras.models import load_model
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
labels = ['male', 'female']

vectors = load_vectors()
max_len = 10

test_name = 'fxp'
training_names = ['fxp', 'twitter', 'training']

df = pd.read_csv(f'datasets/{test_name}_users.csv')
df['name'] = df.name.apply(lambda name: ','.join(name.split(' ')))
test_X, test_Y = turn_to_vectors(df, max_len, vectors)

models = [load_model(f'{name}_lstm.h5') for name in training_names]
for model, name in zip(models, training_names):
    df[name] = ['male' if x == 0 else 'female' for x in
                np.argmax(model.predict(test_X), axis=1)]


def ensemble_label(row):
    d = {'male': 0, 'female': 0}
    d[row['fxp']] += 1
    d[row['twitter']] += 1
    d[row['training']] += 1
    return max(d.items(), key=operator.itemgetter(1))[0]


df['ensemble_label'] = df.apply(ensemble_label, axis=1)

y_true = list(df['gender'])
y_pred = list(df['ensemble_label'])
cm = confusion_matrix(y_true, y_pred, labels)
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
