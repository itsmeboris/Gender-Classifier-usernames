from utility import *
# data set to check accuracy on
test_name = 'okcupid'
# load the dataframe
df = load_split_data(test_name)
# split the data into vectors
test_X, test_Y = turn_to_vectors(df, max_len, vectors)
# load all of the models
models = load_all_models(training_names)
# predict using each model
for model, name in zip(models, training_names):
    # save in the corresponding column
    df[name] = get_labels(model.predict(test_X, verbose=1))
    # plot the results and confusion matrix
    plot_test(df, 'gender', name)
# perform ensemble of all the data
df['ensemble'] = df.apply(ensemble_label, axis=1)
# plot the results and confusion matrix
plot_test(df, 'gender', 'ensemble')
# load the stacked model
stacked_model = load_all_models(['stacked_ensemble'])[0]
# predict using the stacked model
df['stacked'] = get_labels(predict_stacked_model(stacked_model, test_X))
# plot the results and confusion matrix
plot_test(df, 'gender', 'stacked')