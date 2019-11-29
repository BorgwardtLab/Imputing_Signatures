import pandas as pd
from IPython import embed
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', default='results')
args = parser.parse_args()
path = args.path
result = os.listdir(path)

result_dfs = [] # list of dfs to concatenate
for filename in result:
    if 'gathered_results.csv' in filename:
        continue
    filepath = os.path.join(path, filename)
    df = pd.read_csv(filepath)
    result_dfs.append(df)
full_df = pd.concat(result_dfs)
#full_df.to_csv(path + '/gathered_results.csv', index=False)

df = full_df[full_df['use_subsampling'] == False]
columns = ['method', 'dataset', 'test_accuracy']
df = df[columns]

new_columns = ['classifier_name', 'dataset_name', 'accuracy']

col_dict = {}
for i,_ in enumerate(columns):
    col_dict[columns[i]] = new_columns[i]
df = df.rename(columns=col_dict)

df.to_csv(path + '/gathered_results.csv', index=False)

table = pd.pivot_table(df, values='accuracy', index=['dataset_name'], columns=['classifier_name'])
table.to_csv(path  + '/pivoted_results.csv')

embed()

