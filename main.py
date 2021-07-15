from sklearn.model_selection import train_test_split

from cnn import VectorCnn
from stack_ensemble import StackedEnsemble
from utility import *

epochs = 200
models = model_types
model_name = 'gru_best'
for train_name in all_sets:
    print(f'Starting to train on {train_name}')
    models = []
    df = load_split_data(train_name, debug=False, split=True)
    train, test = train_test_split(df, train_size=0.8)
    cnn = VectorCnn(f'{train_name}_{model_name}', epochs, debug=False)
    cnn.build_model(model_name)
    cnn.fit_model(train, test)
    models.append(cnn)
    save_best_model(models, train_name + f'_{model_name}')

for test_name in all_sets[3:]:
    models = []
    print(f'Starting to train stacked ensemble')
    df = load_split_data(test_name, debug=False, split=True)
    train, test = train_test_split(df, train_size=0.8)

    stacked = StackedEnsemble([set + f'_{model_name}' for set in all_sets], f'stacked_ensemble_{test_name}_{model_name}', epochs, debug=False)
    stacked.build_model()
    stacked.fit_model(train, test)
    models.append(stacked)
    save_best_model(models, f'stacked_ensemble_{test_name}_{model_name}')
