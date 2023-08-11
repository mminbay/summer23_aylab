import numpy as np
import pandas as pd
import os
import time
import multiprocessing as mp
import logging
import sys
from multiprocessing import Pool, Manager

OUTPUT_DIR = '/datalake/AyLab/depression_study/depression_snp_data/dominant_model'
DATA_DIR = '/datalake/AyLab/depression_study/depression_snp_data/raw_data'
DEPRESSION_PATH = '/home/mminbay/summer_research/summer23_aylab/data/depression_data.csv'

logging.basicConfig(filename= os.path.join('/home/mminbay/summer_research/summer23_aylab', 'genetic_model.log'), encoding='utf-8', level=logging.DEBUG)

def dominant_model(value):
    if value == 0:
        return 0
    return 1

def recessive_model(value):
    if value == 2:
        return 1
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception('What the hell')
    start = time.time()

    file = sys.argv[1]
    model = sys.argv[2]

    logging.info('APPLYING {} MODEL TO {}, start: '.format(model, file, time.ctime()))
    
    df = pd.read_csv(os.path.join(DATA_DIR, file), index_col = 0)
    
    if model == 'd':
        vectorized_func = np.vectorize(dominant_model)
        prefix = 'dominant_'
    elif model == 'r':
        vectorized_func = np.vectorize(recessive_model)
        prefix = 'recessive_'

        
    cols = []
    cols.append(df['PHQ9_binary'])
    cols.append(df['Sex'])
    cols.append(df['ID_1'])
    logging.info('{} columns in total'.format(str(len(df.columns))))
    for col in df.drop(columns = ['PHQ9_binary', 'Sex', 'ID_1']):
        cols.append(pd.Series(vectorized_func(df[col]), name = col))
        logging.info('Processed a column')  
    conv = time.time()
    logging.info('Applied model in ' + str(conv - start))
    
    result = pd.concat(cols, axis = 1)
    # result['ID_1'] = df.index.astype(int)
    
    # depp_df = pd.read_csv(DEPRESSION_PATH, usecols = ['ID_1', 'PHQ9_binary', 'Sex'])
    # merge = result.merge(depp_df, how = 'inner', on = 'ID_1')
    result.to_csv(os.path.join(OUTPUT_DIR, prefix + file))
    end = time.time()
    logging.info('Merged in ' + str(end - conv))
    
    logging.info('Finished in ' + str(end - start) + '\n')
    

    


