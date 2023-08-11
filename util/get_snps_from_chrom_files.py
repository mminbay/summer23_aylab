import pandas as pd
import os

CHROMS_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/FAULTY_dominant_model/'
SELECT_PATH = '/home/mminbay/summer_research/summer23_aylab/data/dom_overall/feat_select/chi2_top250.csv'


def main():
    select_snps = pd.read_csv(SELECT_PATH, index_col = 0)['SNP'].tolist()
    files = [file for file in os.listdir(CHROMS_DIR) if '.csv' in file]
    result = pd.DataFrame()
    first = pd.read_csv(os.path.join(CHROMS_DIR, files[0]), index_col = 0).sort_values('ID_1')
    result['ID_1'] = first['ID_1']
    result['Sex'] = first['Sex']
    result['PHQ9_binary'] = first['PHQ9_binary']

    columns_set = set(first.columns.tolist())
    for snp in select_snps:
        if snp in columns_set:
            result[snp] = first[snp]

    for file in files[1:]:
        df = pd.read_csv(os.path.join(CHROMS_DIR, file), index_col = 0).sort_values('ID_1')
        columns_set = set(df.columns.tolist())
        for snp in select_snps:
            if snp in columns_set:
                result[snp] = df[snp]


    result.to_csv(os.path.join(SELECT_PATH, '250snps_overall.csv'))

if __name__ == '__main__':
    main()