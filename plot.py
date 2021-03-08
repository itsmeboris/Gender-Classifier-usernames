import re

import matplotlib.pyplot as plt
import turicreate as tc
from tqdm import tqdm

tqdm.pandas()

data_set = 'training'

path_to_train = f'{data_set}_users.csv'

users = tc.SFrame.read_csv(path_to_train, delimiter=',')
users = users.rename({'user_name': 'name'})


def parse_df(sf):
    l = {'name': [], 'gender': [], 'special_chars': [], 'length': [], 'numbers': []}
    for i in tqdm(range(len(sf))):
        try:
            row = sf[i]
            user_name = row["name"].lower()
            l['name'].append(user_name)
            l['gender'].append(row["gender"])
            l['special_chars'].append(len(re.findall(r'[^a-z ]', user_name)))
            l['length'].append(len(user_name))
            l['numbers'].append(len(re.findall(r'[0-9]', user_name)))
        except:
            pass
    return tc.SFrame(l)

users = parse_df(users)
df = users.to_dataframe()
df['gender'] = df['gender'].map(str.lower)
df.columns = ['gender', 'length', 'name', 'numbers', 'special']
pivot = df.pivot_table(index='name', columns='gender')
df.to_csv()

describe = pivot.describe()
describe.to_csv(f'{data_set}.csv')

g = df.groupby('gender')

axis = g.boxplot()
for gender, axes in zip(['female', 'male'], axis):
    axes.set_title(f'{gender} Box Plot', fontsize=14, pad=20)
    axes.set_ylabel('Number of characters', fontsize=14, labelpad=0)
    axes.set_xlabel('Features', fontsize=14)
    axes.tick_params(axis='both', which='major', labelsize=10)
plt.show()