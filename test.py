import warnings
from utility import *

warnings.filterwarnings("ignore")

results = {}
for model_name in model_types:
    # load the stacked models
    stacked_models = load_all_models([f'stacked_ensemble_{name}_{model_name}' for name in all_sets])
    # test_name = 'twitter'
    for test_name in all_sets:
        results[test_name] = {}
        print(f'Testing on: {test_name}')
        print('-' * 80)
        # load the dataframe
        df = load_split_data(test_name)
        # split the data into vectors
        test_X, test_Y = turn_to_vectors(df, max_len, vectors)
        # load all of the models
        models = load_all_models([set + f'_{model_name}' for set in all_sets])
        # predict using each model
        for model, name in zip(models, all_sets):
            # save in the corresponding column
            df[name] = get_labels(
                model.predict(vec_generator(df, max_len, vectors, batch_size), batch_size, verbose=0,
                              steps=(len(df) // batch_size) + 1))
            # plot the results and confusion matrix
            results[test_name][name] = plot_test(df, 'gender', name)
        # perform ensemble of all the data
        df['ensemble'] = df.apply(ensemble_label, axis=1)
        # plot the results and confusion matrix
        results[test_name]['ensemble'] = plot_test(df, 'gender', 'ensemble')
        for stacked_model, name in zip(stacked_models, all_sets):
            # predict using the stacked model
            try:
                df[f'stacked_{name}'] = get_labels(
                    stacked_model.predict(vec_generator(df, max_len, vectors, batch_size, copies=5), batch_size,
                                          verbose=0,
                                          steps=(len(df) // batch_size) + 1))
                # plot the results and confusion matrix
                accuracy = plot_test(df, 'gender', f'stacked_{name}')
            except:
                accuracy = np.nan
                print(f'The accuracy of the classifier stacked_{name} is: {accuracy}')
            results[test_name][f'stacked_{name}'] = accuracy
        try:
            df['ensemle_stacked'] = df.apply(lambda row: ensemble_label(row, 'stacked_'), axis=1)
            accuracy = plot_test(df, 'gender', 'ensemle_stacked')
        except:
            accuracy = np.nan
        results[test_name]['ensemle_stacked'] = accuracy
        print('-' * 80)
    df = pd.DataFrame(results)
    df.to_csv(f'results_{model_name}.csv')
    print(df)
