from cnn import VectorCnn
from stack_ensemble import StackedEnsemble
from utility import *
from tqdm import tqdm

# test_name = 'twitter'
epochs = 70
for train_name in training_names:
    print(f'Starting to train on {train_name}')
    models = {}
    for i in tqdm(range(10)):
        df = load_split_data(train_name, False)
        train, test = train_test_split(df, train_size=0.8)
        cnn = VectorCnn(f'{train_name}_{i}', epochs)
        cnn.build_model()
        cnn.fit_model(train, test)
        models[i] = cnn
    save_best_model(models, train_name)

for test_name in training_names:
    models = {}
    print(f'Starting to train stacked ensemble')
    for i in tqdm(range(1)):
        df = load_split_data(test_name, False)
        train, test = train_test_split(df, train_size=0.8)

        stacked = StackedEnsemble(all_sets, f'stacked_ensemble_{test_name}_{i}', epochs)
        # stacked = StackedEnsemble(list(set(all_sets) - {test_name}), f'stacked_ensemble_{i}', epochs)
        stacked.build_model()
        stacked.fit_model(train, test)
        models[i] = stacked
    save_best_model(models, f'stacked_ensemble_{test_name}')
