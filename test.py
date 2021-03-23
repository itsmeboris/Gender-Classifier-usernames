import warnings
from utility import *

warnings.filterwarnings("ignore")

# test_name = 'twitter'
# load the stacked models
stacked_models = load_all_models([f'stacked_ensemble_{name}' for name in ['fxp', 'twitter', 'okcupid', 'anime']])
for test_name in training_names:
    print(f'Testing on: {test_name}')
    print('-' * 80)
    # load the dataframe
    df = load_split_data(test_name)
    # split the data into vectors
    test_X, test_Y = turn_to_vectors(df, max_len, vectors)
    # load all of the models
    models = load_all_models(all_sets)
    # predict using each model
    for model, name in zip(models, all_sets):
        # save in the corresponding column
        df[name] = get_labels(model.predict(test_X, verbose=1))
        # plot the results and confusion matrix
        plot_test(df, 'gender', name)
    # perform ensemble of all the data
    df['ensemble'] = df.apply(ensemble_label, axis=1)
    # plot the results and confusion matrix
    plot_test(df, 'gender', 'ensemble')
    for stacked_model, name in zip(stacked_models, ['fxp', 'twitter', 'okcupid', 'anime']):
        # predict using the stacked model
        df[f'stacked_{name}'] = get_labels(predict_stacked_model(stacked_model, test_X))
        # plot the results and confusion matrix
        plot_test(df, 'gender', f'stacked_{name}')
    print('-' * 80)
