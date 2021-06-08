import warnings
from utility import *

warnings.filterwarnings("ignore")

predict_file_path = 'datasets/email_list.csv'
df = pd.read_csv(predict_file_path)


def vec_generator(df, max_len, vectors, batch_size=64):
    while True:
        for batch in range(0, len(df), batch_size):
            X = []
            trunc_train_name = [str(i)[0:max_len] for i in df.iloc[batch: batch+batch_size].name]
            for i in trunc_train_name:
                tmp = [vectors[j][0] for j in str(i)]
                for k in range(0, max_len - len(str(i))):
                    tmp.append(vectors['%'][0])
                X.append(tmp)
            yield np.asarray(X)


results = {}
for model_name in ['default', 'LSTM_CONV', 'gru_best', 'bid_GRU_bid_LSTM', 'bidirectional_LSTM']:
    # load the stacked models
    stacked_models = load_all_models([f'stacked_ensemble_{name}_{model_name}' for name in all_sets])
    # load the dataframe
    # load all of the models
    models = load_all_models([set + f'_{model_name}' for set in all_sets])
    # predict using each model
    for model, name in zip(models, all_sets):
        # save in the corresponding column
        df[name] = get_labels(model.predict(vec_generator(df, max_len, vectors, batch_size), batch_size,  verbose=1, steps=(len(df) // batch_size) + 1))
        df.to_csv('datasets/email_list_predicted.csv', index=False)
    # perform ensemble of all the data
    df['ensemble'] = df.apply(ensemble_label, axis=1)
    for stacked_model, name in zip(stacked_models, all_sets):
        # predict using the stacked model
        df[f'stacked_{name}'] = get_labels(stacked_model.predict(vec_generator(df, max_len, vectors, batch_size), batch_size,  verbose=1, steps=(len(df) // batch_size) + 1))
        df.to_csv('datasets/email_list_predicted.csv', index=False)
    df['ensemble_stacked'] = df.apply(lambda row: ensemble_label(row, 'stacked_'), axis=1)
    df.to_csv('datasets/email_list_predicted.csv', index=False)

df.to_csv('datasets/email_list_predicted.csv', index=False)
