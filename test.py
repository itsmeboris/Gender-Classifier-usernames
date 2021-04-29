import warnings
from utility import *

warnings.filterwarnings("ignore")

results = {}
# test_name = 'twitter'
# load the stacked models
stacked_models = load_all_models([f'stacked_ensemble_{name}' for name in ['fxp', 'twitter', 'okcupid', 'anime']])
for test_name in all_sets:
    results[test_name] = {}
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
        accuracy = plot_test(df, 'gender', name)
        results[test_name][name] = accuracy
    # perform ensemble of all the data
    df['ensemble'] = df.apply(ensemble_label, axis=1)
    # plot the results and confusion matrix
    accuracy = plot_test(df, 'gender', 'ensemble')
    results[test_name]['ensemble'] = accuracy
    for stacked_model, name in zip(stacked_models, ['fxp', 'twitter', 'okcupid', 'anime']):
        # predict using the stacked model
        try:
            df[f'stacked_{name}'] = get_labels(predict_stacked_model(stacked_model, test_X))
            # plot the results and confusion matrix
            accuracy = plot_test(df, 'gender', f'stacked_{name}')
        except MemoryError:
            accuracy = np.nan
            print(f'The accuracy of the classifier stacked_{name} is: {accuracy}')
        results[test_name][f'stacked_{name}'] = accuracy
    try:
        df['ensemle_stacked'] = df.apply(lambda row: ensemble_label(row, 'stacked_'), axis=1)
        accuracy = plot_test(df, 'gender', 'ensemle_stacked')
    except KeyError:
        accuracy = np.nan
    results[test_name]['ensemle_stacked'] = accuracy
    print('-' * 80)
df = pd.DataFrame(results)
df.to_csv('results.csv')
print(df)
