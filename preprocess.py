import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

tqdm.pandas()

# df = df.dropna()
# df = df[['name', 'gender']]
# set(' '.join([str(i) for i in df.name])), len(set(' '.join([str(i) for i in df.name])))
# df['gender'] = df['gender'].map(str)
# df.name = df.name.map(str)
# df.gender = df.gender.map(str.lower)
# df = df[df.gender != 'nan']
# df['name'] = df['username'].apply(lambda name: name.replace('ś', 's')
#                                   .replace('ō', 'o').replace('ý', 'y').replace('ü', 'u')
#                                   .replace('ö', 'o').replace('ó', 'o').replace('ñ', 'n')
#                                   .replace('î', 'i').replace('ë', 'e').replace('é', 'e')
#                                   .replace('ç', 'c').replace('æ', 'ae').replace('ã', 'a')
#                                   .replace('â', 'a').replace('á', 'a').replace('à', 'a')
#                                   .replace('â', 'a').replace('á', 'a').replace('à', 'a')
#                                   .replace('ß', 'b').replace('Ü', 'U').replace('Ó', 'O')
#                                   .replace('í', 'i').replace(' ', ',').replace('%40', '@')
#                                   .replace('\x80', '').replace('\x82', '').replace('\x83', '')
#                                   .replace('\x86', '').replace('\x8e', '').replace('\x92', '')
#                                   .replace('\x93', '').replace('\x94', '').replace('\x97', '')
#                                   .replace('\x98', '').replace('\x99', '').replace('\x9d', '').replace('\xad',
#                                                                                                        '').replace(
#     '±', '+-')
#                                   .replace('\xa0', '').replace('¡', 'i').replace('Â', 'A').replace('¶', '')
#                                   .replace('Ð', 'D').replace('¤', '').replace('¦', '').replace('§', 's').replace('©',
#                                                                                                                  'c')
#                                   .replace('Ђ', '').replace('Ѓ', '').replace('Ѕ', 'S').replace('І', 'I').replace('Ј',
#                                                                                                                  'J')
#                                   .replace('Ў', 'y').replace('В', 'B').replace('Г', '').replace('Р', 'P').replace('С',
#                                                                                                                   'C')
#                                   .replace('б', '').replace('в', 'B').replace('п', '').replace('ѓ', '').replace('є',
#                                                                                                                 'e')
#                                   .replace('ї', 'i').replace('љ', '').replace('њ', '').replace('ќ', 'k').replace('ў',
#                                                                                                                  'c')
#                                   .replace('–', '-').replace('—', '-').replace('’', '').replace('‚', '').replace('“',
#                                                                                                                  'c')
#                                   .replace('†', '').replace('…', '').replace('›', '>').replace('™', 'tk').replace('”',
#                                                                                                                   '')
#                                   .replace('½', '1/2').replace('¿', '?').replace('Ã', 'A').replace('ï', 'i'))
# df[df['gender'].isin(['female', 'male'])].to_csv('datasets/anime_users.csv', index=False)

dic = pd.read_csv('datasets/names_dict.csv')
# df.name = df.name.map(str.lower)


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


# Define formula to parallelize the code on multiple cores
def parallelize_dataframe(df, func, n_cores=8):
    rdf = []
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    rdf.extend(pool.map(func, df_split))
    pool.close()
    pool.join()
    return pd.concat(rdf)


if __name__ == '__main__':
    df = pd.read_csv('datasets/email_pass170k_NIST_metrics.csv', dtype='str')
    a = df.name.progress_apply(get_gender_from_dic_ext) # parallelize_dataframe(df.name, get_gender_from_dic_ext)

    genders = []
    names = []
    for gender, name in a:
        genders.append(gender)
        names.append(name)
    df['gender_by_dict_ext'] = genders
    df['chosen_name'] = names
