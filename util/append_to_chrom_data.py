import pandas as pd
import os

DATA_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/final_data'
files = [file for file in os.listdir(DATA_DIR) if 'csv' in file]
depression_data = pd.read_csv('/home/mminbay/summer_research/summer23_aylab/data/depression_data.csv', usecols = ['ID_1', 'PHQ9_binary', 'Sex'])

for file in files:
    df = pd.read_csv(os.path.join(DATA_DIR, file), index_col = 0)
    df['ID_1'] = df.index.astype(int)
    merge = df.merge(depression_data, how = 'inner', on = 'ID_1')
    merge.to_csv(os.path.join(DATA_DIR, 'final_' + file))