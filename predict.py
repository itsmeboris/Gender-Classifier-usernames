import warnings
from utility import *
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

tqdm.pandas()

warnings.filterwarnings("ignore")

predict_file_path = 'datasets/email_pass170k_NIST_metrics.csv'
df = pd.read_csv(predict_file_path)

dic = pd.read_csv('datasets/names_dict.csv')
dic = dic[dic['gender'].isin(labels)]

words = list(dic['name'])
gender_list = list(dic['gender'])
vectorizer = CountVectorizer(vocabulary=words,
                             binary=True)


def get_gender_sk(sentence):
    bagowords = vectorizer.fit_transform([sentence]).toarray()
    idx_vector = bagowords[0]
    d = {'male': 0, 'female': 0}
    name = {'male': '', 'female': ''}
    for idx, (present, gender) in enumerate(zip(idx_vector, gender_list)):
        if not present:
            continue
        d[gender] += 1
        name[gender] = words[idx] if len(words[idx]) > len(name[gender]) else {}
    if d['male'] > d['female']:
        return 'male', name['male']
    if d['female'] > d['male']:
        return 'female', name['female']
    if len(name['male']) > len(name['female']):
        return 'male', name['male']
    if len(name['female']) > len(name['male']):
        return 'female', name['female']
    return 'unknown', ''


a = df.name.progress_apply(get_gender_sk)
genders = []
names = []

for gender, name in a:
    genders.append(gender)
    names.append(name)

df['gender_by_dict'], df['chosen_name'] = genders, names

print('Starting prediction')
print('-' * 80)
model_name = 'LSTM_CONV'
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

df['gender'] = df.progress_apply(
    lambda row: row['gender_by_dict'] if row['gender_by_dict'] != 'unknown' else row['ensemble_gender'], axis=1)
df.to_csv('datasets/email_pass170k_NIST_metrics_predicted.csv', index=False)
print('-' * 80)
