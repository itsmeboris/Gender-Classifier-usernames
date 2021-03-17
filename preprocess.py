import pandas as pd

df = pd.read_csv('datasets/anime_users.csv', dtype='str')
df = df.dropna()
df = df[['name', 'gender']]
set(' '.join([str(i) for i in df.name])), len(set(' '.join([str(i) for i in df.name])))
df['gender'] = df['gender'].map(str)
df.name = df.name.map(str)
df.gender = df.gender.map(str.lower)
df = df[df.gender != 'nan']
df['name'] = df['name'].apply(lambda name: name.replace('-', '').replace('ś', 's')
                              .replace('ō', 'o').replace('ý', 'y').replace('ü', 'u')
                              .replace('ö', 'o').replace('ó', 'o').replace('ñ', 'n')
                              .replace('î', 'i').replace('ë', 'e').replace('é', 'e')
                              .replace('ç', 'c').replace('æ', 'ae').replace('ã', 'a')
                              .replace('â', 'a').replace('á', 'a').replace('à', 'a')
                              .replace('â', 'a').replace('á', 'a').replace('à', 'a')
                              .replace('ß', 'b').replace('Ü', 'U').replace('Ó', 'O')
                              .replace('í', 'i').replace(' ', ',').replace('%40', '@')
                              .replace('\x80', '').replace('\x82', '').replace('\x83', '')
                              .replace('\x86', '').replace('\x8e', '').replace('\x92', '')
                              .replace('\x93', '').replace('\x94', '').replace('\x97', '')
                              .replace('\x98', '').replace('\x99', '').replace('\x9d', '')
                              .replace('\xa0', '').replace('¡', 'i').replace('Â', 'A')
                              .replace('Ð', 'D'))
df[df['gender'].isin(['female', 'male'])].to_csv('datasets/anime_users.csv', index=False)