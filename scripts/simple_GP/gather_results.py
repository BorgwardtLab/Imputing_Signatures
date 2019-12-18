import pandas as pd
from IPython import embed
import os
import argparse

def intersection(l1, l2):
    l3 = [v for v in l1 if v in l2]
    return l3

parser = argparse.ArgumentParser()

parser.add_argument('--path', default='results')
args = parser.parse_args()
path = args.path
result = os.listdir(path)

result_dfs = [] # list of dfs to concatenate
for filename in result:
    #if 'gathered_results.csv' in filename:
    if any(x in filename for x in ['results', 'singleTrainTest']):        
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

#load official UCR results:
ucr_results = pd.read_csv(path + '/singleTrainTest.csv') 
ucr_results = ucr_results.rename(columns={'Unnamed: 0': 'dataset_name'}) 
#Compare Datasets:
ucr_datasets = ucr_results['dataset_name']
ucr_datasets = ucr_datasets.values.tolist()

my_datasets = table.index.tolist()

joint_datasets = intersection(ucr_datasets, my_datasets)

# get record form for CD plot
ucr_records = ucr_results.melt(id_vars=['dataset_name'], var_name='classifier_name', value_name='accuracy')
table_records = table.reset_index().melt(id_vars=['dataset_name'], var_name='classifier_name', value_name='accuracy')

# subset only joint datasets (for quick testing now):
ucr_records = ucr_records[ucr_records.dataset_name.isin(joint_datasets)]
table_records = table_records[table_records.dataset_name.isin(joint_datasets)]
table_records = table_records[table_records.classifier_name != 'DTW_kNN'] #remove our own dtw-kNN as official results contain them in various ways already..

full_table = pd.concat([table_records, ucr_records])
full_table.to_csv(path + '/full_table_results.csv', index=False)

full_tabular = pd.pivot_table(full_table, values='accuracy', index=['dataset_name'], columns=['classifier_name'])
full_tabular.to_csv(path + '/full_tabular_results.csv', index=True)

embed()

