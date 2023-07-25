import numpy as np
import pandas as pd
import os
import time
import multiprocessing as mp
from multiprocessing import Pool

OUTPUT_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/recessive_model'
DATA_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/final_data'
DEPRESSION_PATH = '/home/mminbay/summer_research/summer23_aylab/data/depression_data.csv'

def recessive_wrapper(data_path, depression_path, output_path):
    t = time.time()
    df = pd.read_csv(data_path, index_col = 0)
    col_names = []
    cols = []
    vectorized_func = np.vectorize(recessive_model)
    for col in df:
        col_names.append(col)
        cols.append(pd.Series(vectorized_func(df[col]), name = col))
    result = pd.concat(cols, axis = 1)
    result['ID_1'] = df.index.astype(int)
    depp_df = pd.read_csv(depression_path, usecols = ['ID_1', 'PHQ9_binary', 'Sex'])
    merge = result.merge(depp_df, how = 'inner', on = 'ID_1')
    merge.to_csv(output_path)
    return time.time() - t

def recessive_model(value):
    if value == 2:
        return 1
    return 0

if __name__ == '__main__': 
    full_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if '.csv' in file and 'final' not in file]
    output_files = [os.path.join(OUTPUT_DIR, 'recessive_' + file) for file in os.listdir(DATA_DIR) if '.csv' in file and 'final' not in file]
    full_files = [os.path.join(DATA_DIR, 'c15_i6.csv')]
    output_files = [os.path.join(OUTPUT_DIR, 'test_c15_i6.csv')]
    tuples = [(full_files[i], DEPRESSION_PATH, output_files[i]) for i in range(len(full_files))]

    with Pool() as pool:
        results = pool.starmap(recessive_wrapper, tuples)

    print(results)
    


