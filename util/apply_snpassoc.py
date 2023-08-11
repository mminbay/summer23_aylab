import pandas as pd
import numpy as np
import os

def pairwise(snp1, snp2):
    if snp1 + snp2 == 2:
        return 1
    return 0

def main():
    df = pd.read_csv('/home/mminbay/summer_research/summer23_aylab/data/male_dom/feat_select/SNPAssoc/for_snpassoc_both.csv', index_col = 0)
    interactions = [
        ('rs139244344', 'rs543118396'),
        ('rs200338172', 'rs757675910')
    ]

    for snp1, snp2 in interactions:
        df['pair:{}-{}'.format(snp1, snp2)] = np.vectorize(pairwise)(df[snp1], df[snp2])

    df.to_csv('/home/mminbay/summer_research/summer23_aylab/data/male_dom/analysis/final_data.csv')

if __name__ == '__main__':
    main()