import warnings
from utility import *

warnings.filterwarnings("ignore")

predict_file_path = 'datasets/email_pass170k_NIST_metrics.csv'
df = pd.read_csv(predict_file_path)

dic = pd.read_csv('datasets/names_dict.csv')


def get_gender_from_dic_ext(x):
    max_len_f = ""
    max_len_m = ""
    count_m = 0
    count_f = 0
    seper = "-.,_"
    if not any([c in x for c in seper]):
        return 'unknown', ""
    for ind, row in dic.iterrows():
        if row['name'] in x:
            if row['gender'] == "female":
                count_f += 1
                if len('name') > len(max_len_f):
                    max_len_f = row['name']
            if row['gender'] == "male":
                count_m += 1
                if len(row['name']) > len(max_len_m):
                    max_len_m = row['name']
    if count_m > count_f:
        return "male", max_len_m
    elif count_f > count_m:
        return "female", max_len_f
    elif len(max_len_m) > len(max_len_f):
        return "male", max_len_m
    elif len(max_len_f) > len(max_len_m):
        return "female", max_len_f
    return "unknown", ""


print('Starting prediction')
print('-' * 80)
model_name = 'LSTM_CONV'
# # load the stacked models
# stacked_models = load_all_models([f'stacked_ensemble_{name}_{model_name}' for name in all_sets])
# load all of the models
models = load_all_models([set + f'_{model_name}' for set in all_sets])
# predict using each model
for model, name in zip(models, all_sets):
    # save in the corresponding column
    df[f'{model_name}_{name}'] = get_labels(
        model.predict(vec_generator(df, max_len, vectors, batch_size), batch_size, verbose=0,
                      steps=(len(df) // batch_size) + 1))
    print(f'finished predicting: {model_name}_{name}')
# perform ensemble of all the data
df[f'ensemble_gender'] = df.progress_apply(lambda row: ensemble_label(row, f'{model_name}_'), axis=1)
print('predicted ensemble')
# for stacked_model, name in zip(stacked_models, all_sets):
#     # predict using the stacked model
#     df[f'stacked_{model_name}_{name}'] = get_labels(
#         stacked_model.predict(vec_generator(df, max_len, vectors, batch_size, copies=5), batch_size, verbose=0,
#                               steps=(len(df) // batch_size) + 1))
#     print(f'finished predicting: stacked_{model_name}_{name}')
# df[f'ensemble_stacked_{model_name}'] = df.apply(lambda row: ensemble_label(row, f'stacked_{model_name}_'), axis=1)
# print('predicted stacked ensemble')

a = df.name.progress_apply(get_gender_from_dic_ext)

genders = []
names = []
for gender, name in a:
    genders.append(gender)
    names.append(name)
df['gender_by_dict'] = genders
df['chosen_name'] = names

df['gender'] = df.progress_apply(
    lambda row: row['gender_by_dict_ext'] if row['gender_by_dict_ext'] != 'unknown' else row['ensemble_gender'], axis=1)
df.to_csv('datasets/email_pass170k_NIST_metrics_predicted.csv', index=False)
print('-' * 80)

df.to_csv('datasets/email_pass170k_NIST_metrics_predicted.csv', index=False)
