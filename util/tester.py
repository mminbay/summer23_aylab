import os
import pandas as pd
import multiprocessing as mp
import logging

DOM_DIR = '/datalake/AyLab/depression_study/depression_snp_data/dominant_model'
RAW_DIR = '/datalake/AyLab/depression_study/depression_snp_data/raw_data'

def wrapper(path):
    df = pd.read_csv(path, index_col = 0)
    columns = df.filter(regex = '\.[0-9]$').columns
    multiple_columns = {col.split('.')[0] for col in columns}
    logging.info('{} has {} multiplicities: {}'.format(os.path.basename(path), str(len(multiple_columns)), str(multiple_columns)))
    for col in multiple_columns:
        df[col + '_any'] = df.filter(regex = col).any(axis = 1).astype(int)
    df.to_csv(path)
    # logging.info('{} has Sex: {}'.format(os.path.basename(path), 'Sex' in df.columns))
    # logging.info('{} has PHQ9_binary: {}'.format(os.path.basename(path), 'PHQ9_binary' in df.columns))
    # logging.info('{} has ID_1: {}'.format(os.path.basename(path), 'ID_1' in df.columns))
    # logging.info('{} has na: {}'.format(os.path.basename(path), str(df.isna().sum().sum())))
    # logging.info('{} has 2: {}'.format(os.path.basename(path), str((df == 2).sum().sum())))

def main():
    logging.basicConfig(filename= '/home/mminbay/tester.log', encoding='utf-8', level=logging.DEBUG)
    paths = [os.path.join(DOM_DIR, file) for file in os.listdir(DOM_DIR) if '.csv' in file]
        
    with mp.Pool() as pool:
        pool.map(wrapper, paths)
    # map = {os.path.basename(paths[i]): results[i] for i in range(len(paths))}
    # sum = 0
    # for key1 in map:
    #     sum += len(map[key1])
    #     for key2 in map:
    #         logging.info('{} has {} common SNPs with {}'.format(key1, str(len(map[key1].intersection(map[key2]))), key2))

    
if __name__ == '__main__':
    main()