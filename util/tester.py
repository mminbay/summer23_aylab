import os
import pandas as pd
import multiprocessing as mp
import logging

DOM_DIR = '/datalake/AyLab/depression_study/depression_snp_data/dominant_model'
RAW_DIR = '/datalake/AyLab/depression_study/depression_snp_data/raw_data'

def wrapper(raw, dom):
    data = pd.read_csv(raw, index_col = 0)
    print(file, len(data))
    freqs = {}
    m = data.filter(regex = '\.\d$').columns
    m2 = [s[:s.index('.')] for s in m]
    for col in m2:
        this_snp = (data.filter(regex = col) != 0).any(axis = 1).sum()
        freqs[col] = this_snp
    data = pd.read_csv(dom, index_col = 0)
    for col in freqs.keys():
        print(col, data[col + '_any'] == freqs[col])
    # logging.info('{} has Sex: {}'.format(os.path.basename(path), 'Sex' in df.columns))
    # logging.info('{} has PHQ9_binary: {}'.format(os.path.basename(path), 'PHQ9_binary' in df.columns))
    # logging.info('{} has ID_1: {}'.format(os.path.basename(path), 'ID_1' in df.columns))
    # logging.info('{} has na: {}'.format(os.path.basename(path), str(df.isna().sum().sum())))
    # logging.info('{} has 2: {}'.format(os.path.basename(path), str((df == 2).sum().sum())))

def main():
    logging.basicConfig(filename= '/home/mminbay/tester.log', encoding='utf-8', level=logging.DEBUG)
    raw_files = [file for file in os.listdir(RAW_DIR) if 'additional' in file and 'alleles' not in file]
    paths = [(os.path.join(RAW_DIR, file), os.path.join(DOM_DIR, 'dominant_' + file)) for file in raw_files]
        
    with mp.Pool() as pool:
        pool.starmap(wrapper, paths)
    # map = {os.path.basename(paths[i]): results[i] for i in range(len(paths))}
    # sum = 0
    # for key1 in map:
    #     sum += len(map[key1])
    #     for key2 in map:
    #         logging.info('{} has {} common SNPs with {}'.format(key1, str(len(map[key1].intersection(map[key2]))), key2))

    
if __name__ == '__main__':
    main()