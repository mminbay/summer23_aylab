import os
import numpy as np
import pandas as pd

DATA_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/final_data'
files = os.listdir(DATA_DIR)

final_files = [file for file in files if 'csv' in file]

final_table = pd.DataFrame()
indices = pd.read_csv(os.path.join(DATA_DIR, final_files[0]), usecols = [0], index_col = 0)
final_table.index = indices.index
final_table.sort_index(inplace = True)
del indices


for file in final_files:
    df = pd.read_csv(os.path.join(DATA_DIR, file), index_col = 0)
    df.sort_index(inplace = True)
    if list(df.index) != list(final_table.index):
        raise Exception('Holy macaroni, the ID\'s are different')
    final_table = pd.concat([final_table, df], axis = 1)
    del df

final_table['ID_1'] = final_table.index.astype(int)
final_table.to_csv(os.path.join(DATA_DIR, 'merged_1/4.csv'))