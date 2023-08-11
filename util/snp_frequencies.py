import pandas as pd
import os

DATA_DIR = '/datalake/AyLab/depression_snp_data/dominant_model/'
chi2_df = pd.read_csv('/home/mminbay/summer_research/summer23_aylab/data/dom_overall/feat_select/chi2_all.csv', index_col = 0)
chi2_df.drop_duplicates(subset = 'SNP', inplace = True)
columns = chi2_df['SNP'].tolist()


files = [file for file in os.listdir(DATA_DIR) if '.csv' in file]
result_dfs = []

for file in files:
    print('Reading {}'.format(file))
    df = pd.read_csv(os.path.join(DATA_DIR, file), index_col = 0)
    result = pd.DataFrame()
    result['SNP'] = df.columns
    result['Frequency'] = 0
    result.drop(result[result['SNP'].isin(['PHQ9_binary', 'ID_1', 'Sex'])].index, inplace = True)
    for snp in result['SNP']:
        try:
            result.loc[result['SNP'] == snp, 'Frequency'] = df[snp].value_counts()[1]
        except:
            continue
    result_dfs.append(result)

cc = pd.concat(result_dfs)
chi2_df = chi2_df.merge(cc, on = 'SNP')

chi2_df.to_csv('/home/mminbay/summer_research/summer23_aylab/data/dom_overall/feat_select/chi2_all_freq.csv')
