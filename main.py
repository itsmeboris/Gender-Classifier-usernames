from cnn import VectorCnn
from stack_ensemble import StackedEnsemble
from utility import *

test_name = 'twitter'
epochs = 70
for train_name in training_names:
    df = load_split_data(train_name, False)
    train, test = train_test_split(df, train_size=0.8)

    cnn = VectorCnn(train_name, epochs)
    cnn.build_model()
    cnn.fit_model(train, test)
    cnn.show_history()

df = load_split_data(test_name, False)
train, test = train_test_split(df, train_size=0.8)

stacked = StackedEnsemble(list(set(all_sets) - {test_name}), test_name, epochs)
stacked.build_model()
stacked.fit_model(train, test)
stacked.show_history()
