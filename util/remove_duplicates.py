import pandas as pd
import os

OUTPUT_DIR = '/datalake/AyLab/depression_snp_data/raw_data'
OLD_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/final_data'
COL_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/dominant_model'

old_files = [file for file in os.listdir(OLD_DIR) if '.csv' in file]

for file in old_files:
    new_file = '_'.join(['dominant'] + (file.split('_'))[1:])
    columns_to_use = pd.read_csv(os.path.join(COL_DIR, new_file), index_col = 0, nrows = 1).columns.tolist()
    old_file = pd.read_csv(os.path.join(OLD_DIR, file), usecols = columns_to_use)
    old_file.to_csv(os.path.join(OUTPUT_DIR, file))