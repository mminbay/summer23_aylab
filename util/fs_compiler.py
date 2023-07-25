import pandas as pd
import os

FS_DIR = '/home/mminbay/summer_research/summer23_aylab/data/dom_overall/feat_select/'

def main():
    # dirs = os.listdir(FS_DIR)
    # all_results = [pd.read_csv(os.path.join(FS_DIR, dir, 'name.csv'), index_col = 0) for dir in dirs if '.' not in dir]
    # print(len(all_results))

    # merged = pd.concat(all_results)
    # chi2_results = merged[['average_chi2', 'nan_chi2', 'total_chi2', 'SNP']]
    # chi2_todrop = chi2_results.loc[chi2_results['nan_chi2'] >= 2].index
    # chi2_results.drop(labels = chi2_todrop, inplace = True)
    # chi2_results.sort_values(['average_chi2'], ascending = False, inplace = True)
    # chi2_results.to_csv(os.path.join(FS_DIR, 'chi2_all.csv'))

    # infogain_results = merged[['average_infogain', 'nan_infogain', 'total_infogain', 'SNP']]
    # infogain_results.sort_values(['average_infogain'], ascending = False, inplace = True)
    # infogain_results.to_csv(os.path.join(FS_DIR, 'infogain_all.csv'))

    chi2_all = pd.read_csv(os.path.join(FS_DIR, 'chi2_all.csv'), index_col = 0)
    chi2_all.drop_duplicates(subset = 'SNP', inplace = True)
    chi2_all['chi2_avg_rank'] = chi2_all['average_chi2'].rank(ascending = False)
    cutoff = chi2_all.iloc[[249]]['chi2_avg_rank'].item()
    top250 = chi2_all[chi2_all['chi2_avg_rank'] <= cutoff]
    top250.to_csv(os.path.join(FS_DIR, 'chi2_top250.csv'))

    

if __name__ == '__main__':
    main()