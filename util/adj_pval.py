import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import os

def adjust_pval(path):
    data = pd.read_csv(path, index_col = 0)
    cols = data.filter(regex = 'p_val').columns
    
    for i in range(len(cols)):
        data['adj_{}'.format(str(i))] = fdrcorrection(data[cols[i]])[1]
    
    dir = os.path.dirname(path)
    name = os.path.basename(path).split('.')[0] + '_adjusted.csv'
    data.to_csv(os.path.join(dir, name))

paths = [
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/female/feat_select/000_first_round_results/female_bin_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/female/feat_select/000_first_round_results/female_cont_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/female/feat_select/000_first_round_results/female_ttest_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/male/feat_select/000_first_round_results/male_bin_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/male/feat_select/000_first_round_results/male_cont_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/male/feat_select/000_first_round_results/male_ttest_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/overall/feat_select/000_first_round_results/overall_bin_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/overall/feat_select/000_first_round_results/overall_cont_all.csv',
    '/home/mminbay/summer_research/summer23_aylab/data/frequent_snps_analysis/overall/feat_select/000_first_round_results/overall_ttest_all.csv',
]

for path in paths:
    adjust_pval(path)