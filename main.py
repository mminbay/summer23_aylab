from feature_selector import FeatureSelector
from load_data import DataLoader
from statistical_analysis import Stat_Analyzer
from classifiers import ClassifierHelper
from utils import *
import pandas as pd
import numpy as np
import os

'''
Ultimate main file. This should "hopefully" do the entire study for you. Play around with the paths below to your liking.
'''

### OVERALL CONSTANTS --- THIS IS WHERE THE OUTPUTS WILL GENERALLY BE DUMPED TO
HOME_DIR = '/home/mminbay/summer_research/depression_study/'
RUNS_DIR = os.path.join(HOME_DIR, 'runs')
GROUPS = ['male', 'female', 'overall']

### DATA LOADER CONSTANTS
CLINICAL_FACTORS = [
    ("Sex", 31, "binary"),
    ("Age", 21022, "continuous"),
    ("Chronotype", 1180, "continuous"),
    ("Sleeplessness/Insomnia", 1200, "continuous"),
    ("TSDI", 22189, 'continuous'),
    ("Health_Score", 2178, 'continuous')
]
POSITIVE_CLINICAL_FACTORS = ['Chronotype', 'Sleeplessness/Insomnia', 'Health_Score']
DEPRESSION_DATA_PATH = os.path.join(HOME_DIR, 'depression_data.csv')
INTERVALS_PATH = os.path.join(HOME_DIR, 'all_intervals.txt')
BGEN_DIR = '/datalake/AyLab/data1/ukb_genetic_data/'
BGEN_FORMAT = 'ukb22828_c{}_b0_v3.bgen'
BGEN_SAMPLES_PATH = '/datalake/AyLab/data1/ukb_genetic_data/ukb22828_c1_b0_v3_s487159.csv'
GENETIC_RAW_OUTPUT_DIR = '/datalake/AyLab/depression_study/depression_snp_data/raw_data/'
DOMINANT_MODEL_DIR = '/datalake/AyLab/depression_study/depression_snp_data/dominant_model/'

### FEAT SELECT CONSTANTS
FS_DIR = os.path.join(RUNS_DIR, '{}', 'feat_select')
FIRST_ROUND_BINARY_FORMAT = '{}_binary'
FIRST_ROUND_CONTINUOUS_FORMAT = '{}_continuous'
FREQ_THRESHOLDS = [50, 50, 100] # appear in same order as GROUPS
SNPASSOC_FREQ_THRESHOLDS = [10, 10, 20] # appear in same order as GROUPS
SNPASSOC_PATH = '/home/mminbay/summer_research/summer23_aylab/Rscripts/SNPAssoc.R'

def main():
    ### DATA LOADING --- these steps are already done! 
    # Assuming your ukb dataset exists, and your .ukbb_paths.py file has been correctly configured
    dl = DataLoader(
        genetics_folder = BGEN_DIR,
        genetics_format = BGEN_FORMAT,
        imputed_ids_path = BGEN_SAMPLES_PATH,
        out_folder = GENETIC_RAW_OUTPUT_DIR
    )

    dl.create_table('factors', CLINICAL_FACTORS)
    dl.calcPHQ9('PHQ9_scores', binary_cutoff = 10)
    factors_table = dl.get_table('factors')
    phq9_table = dl.get_table('PHQ9_scores')
    data = factors_table.merge(phq9_table, on = 'ID_1')
    # Drop negative and Na values from these columns, as they mean no data or prefer not to answer
    data = data.dropna()
    data = data[(data[POSITIVE_CLINICAL_FACTORS] >= 0).all(axis = 1)]
    # TODO: normalize Age and TSDI
    data.to_csv(DEPRESSION_DATA_PATH)

    # Assuming your chroms and intervals file exists and is properly formatted
    dl.load_chroms_and_intervals(INTERVALS_PATH)
    # This step takes VERY LONG if run sequentially. You might want to split it up to different jobs with the 'keep' and 'ignore' parameters
    dl.get_imputed_from_intervals_for_ids(
        pd.read_csv(DEPRESSION_DATA_PATH, index_col = 0),
        extra = 6000,
        table_name = 'c{}_depression_6000extra.csv',
        export = True,
        keep_track_as = 'path',
        use_list = False,
        get_alleles = True
    )
    ### TODO: DOMINANT MODEL (already done, but will add working code for it here)

    ### ACTUAL RUNS

    ### FIRST ROUND FEATURE SELECTION --- LEFT HAND SIDE (no data split)
    ### THIS ASSUMES THAT CHROM FILES ONLY HAVE 'ID_1' AND SNP INFO
    paths = [os.path.join(DOMINANT_MODEL_DIR, path) for path in os.listdir(DOMINANT_MODEL_DIR) if '.csv' in path]
    for i in range(len(GROUPS)):
        group = GROUPS[i]
        freq_threshold = FREQ_THRESHOLDS[i]
        
        first_round_dir = os.path.join(FS_DIR.format(group), 'first_round')
        if not os.path.exists(first_round_dir):
            os.makedirs(first_round_dir)

        bootstrap_loc = ''

        for j in range(len(paths)):
            chrom_name = re.search('(?<=_)c[0-9]+(_i[0-9]+)?', paths[j]).group(0) # regex that grabs c{}, where {} is replaced by chrom number
            chrom_data = pd.read_csv(paths[j], index_col = 0)
            binary_fs_data = merge_for_fs(chrom_data, DEPRESSION_DATA_PATH, sex = group, outcome = 'bin')
            continuous_fs_data = merge_for_fs(chrom_data, DEPRESSION_DATA_PATH, sex = group, outcome = 'cont')

            # for first run, bootstrap must be randomized
            if j == 0:
                # binary feature selection
                bin_output_dir = os.path.join(first_round_dir, FIRST_ROUND_BINARY_FORMAT.format(chrom_name))
                fselect = FeatureSelector(
                    binary_fs_data,
                    bin_output_dir
                )

                fselect.bootstrapped_feat_select(
                    freq_threshold,
                    3,
                    None,
                    'PHQ9_binary',
                    ['chi2'],
                    [{}],
                    '{}_chi2'.format(group)
                )

                bootstrap_loc = os.path.join(bin_output_dir, '{}_chi2_bootstraps.csv'.format(group))

                # continuous feature selection
                cont_output_dir = os.path.join(first_round_dir, FIRST_ROUND_CONTINUOUS_FORMAT.format(chrom_name))
                fselect = FeatureSelector(
                    continuous_fs_data,
                    cont_output_dir
                )
                bs = fselect.load_bootstraps(bootstrap_loc)

                fselect.bootstrapped_feat_select(
                    freq_threshold,
                    3,
                    None,
                    'PHQ9',
                    ['ttest'],
                    [{}],
                    '{}_ttest'.format(group),
                    bootstraps = bs
                )
            # for all other runs, refer to the generated bootstrap
            else:
                # binary feature selection
                bin_output_dir = os.path.join(first_round_dir, FIRST_ROUND_BINARY_FORMAT.format(chrom_name))
                fselect = FeatureSelector(
                    binary_fs_data,
                    bin_output_dir
                )

                fselect.bootstrapped_feat_select(
                    freq_threshold,
                    3,
                    None,
                    'PHQ9_binary',
                    ['chi2'],
                    [{}],
                    '{}_chi2'.format(group),                    
                    bootstraps = bs
                )

                # continuous feature selection
                cont_output_dir = os.path.join(first_round_dir, FIRST_ROUND_CONTINUOUS_FORMAT.format(chrom_name))
                fselect = FeatureSelector(
                    continuous_fs_data,
                    cont_output_dir
                )
                
                bs = fselect.load_bootstraps(bootstrap_loc)

                fselect.bootstrapped_feat_select(
                    freq_threshold,
                    3,
                    None,
                    'PHQ9',
                    ['ttest'],
                    [{}],
                    '{}_ttest'.format(group),
                    bootstraps = bs
                )
        # Evaluating and compiling first round results --- binary
        compile_fs_results(
            first_round_dir,
            identifier = 'binary',
            out_name = '{}_binary_summary.csv'.format(group)
        )
        binary_summary = pd.read_csv(first_round_dir, '{}_binary_sumary.csv'.format(group), index_col = 0)
        binary_survived = set(binary_summary[(binary_summary.filter(regex = '[0-9]_p_val$') <= 0.05).all(axis = 1)]['SNP'].tolist())
        compile_snps(binary_survived, ['ID_1'], DOMINANT_MODEL_DIR, os.path.join(FS_DIR.format(group), '{}_binary_compiled.csv'.format(group)))

        # Evaluating and compiling first round results --- continuous
        compile_fs_results(
            first_round_dir,
            identifier = 'continuous',
            out_name = '{}_continuous_summary.csv'.format(group)
        )
        continuous_summary = pd.read_csv(first_round_dir, '{}_continuous_sumary.csv'.format(group), index_col = 0)
        continuous_survived = set(binary_summary[(continuous_summary.filter(regex = '[0-9]_p_val$') <= 0.05).all(axis = 1)]['SNP'].tolist())
        compile_snps(continuous_survived, ['ID_1'], DOMINANT_MODEL_DIR, os.path.join(FS_DIR.format(group), '{}_continuous_compiled.csv'.format(group)))

    ### SECOND ROUND FEATURE SELECTION --- Technically this could be part of the same loop, but I thought it would be better to split
    for i in range(len(GROUPS)):
        group = GROUPS[i]
        freq_threshold = FREQ_THRESHOLDS[i]

        second_round_dir = os.path.join(FS_DIR.format(group), 'second_round')

        ### logistic lasso
        log_lasso_data = pd.read_csv(os.path.join(FS_DIR.format(group), '{}_binary_compiled.csv'.format(group)), index_col = 0)

        fselect = FeatureSelector(
            merge_for_fs(log_lasso_data, DEPRESSION_DATA_PATH, group, outcome = 'bin'),
            os.path.join(second_round_dir, 'binary')
        )

        fselect.bootstrapped_feat_select(
            50,
            3,
            None,
            'PHQ9_binary',
            ['lasso'],
            [{
                'outcome': 'discrete',
                'n_selected_features': 50
            }],
            '{}_lasso_binary'.format(group),
        )

        ### evaluate and compile logistic lasso results
        log_lasso_results = pd.read_csv(os.path.join(second_round_dir, 'binary', '{}_lasso_binary.csv'), index_col = 0)
        log_lasso_survived = log_lasso_results[(log_lasso_results.filter(regex = 'score$') > 0).all(axis = 1)]['SNP'].tolist()
        log_lasso_survived.extend(['ID_1', 'PHQ9_binary'])
        log_lasso_final = log_lasso_data[log_lasso_survived]

        bootstrap_loc = os.path.join(second_round_dir, 'binary', '{}_lasso_binary_bootstraps.csv'.format(group))

        ### linear lasso
        lin_lasso_data = pd.read_csv(os.path.join(FS_DIR.format(group), '{}_continuous_compiled.csv'.format(group)), index_col = 0)

        fselect = FeatureSelector(
            merge_for_fs(lin_lasso_data, DEPRESSION_DATA_PATH, group, outcome = 'cont'),
            os.path.join(second_round_dir, 'continuous')
        )

        fselect.bootstrapped_feat_select(
            50,
            3,
            None,
            'PHQ9',
            ['lasso'],
            [{
                'outcome': 'continuous',
                'n_selected_features': 50
            }],
            '{}_lasso_continuous'.format(group),
            bootstraps = fselect.load_bootstraps(bootstrap_loc)
        )

        ### evaluate and compile linear lasso results
        lin_lasso_results = pd.read_csv(os.path.join(second_round_dir, 'continuous', '{}_lasso_continuous.csv'), index_col = 0)
        lin_lasso_survived = lin_lasso_results[(lin_lasso_results.filter(regex = 'score$') > 0).all(axis = 1)]['SNP'].tolist()
        lin_lasso_survived.extend(['ID_1', 'PHQ9'])
        lin_lasso_final = lin_lasso_data[lin_lasso_survived]

        final_results = lin_lasso_final.merge(log_lasso_final, on = 'ID_1', suffixes = ('', '_y'))
        final_results.drop(columns = final_results.filter(regex = '_y').columns, inplace = True)
        final_results = reorder_cols(final_results) # check utils.py
        
        snpassoc_dir = os.path.join(FS_DIR.format(group), 'snpassoc')
        if not os.path.exists(snpassoc_dir):
            os.makedirs(snpassoc_dir)
            
        final_results.to_csv(os.path.join(snpassoc_dir, '{}_for_snpassoc.csv'.format(group)))

    ### SNPAssoc
    for i in range(len(GROUPS)):
        group = GROUPS[i]
        snpassoc_dir = os.path.join(FS_DIR.format(group), 'snpassoc')
        snpassoc_data_path = os.path.join(snpassoc_dir, '{}_for_snpassoc.csv'.format(group))
        snpassoc_wrapper(SNPASSOC_PATH, snpassoc_data_path, 'PHQ9')
        snpassoc_wrapper(SNPASSOC_PATH, snpassoc_data_path, 'PHQ9_binary')

        snpassoc_bin_results_path = os.path.join(snpassoc_dir, '{}_for_snpassoc_bin_snpassoc.csv'.format(group))
        bin_interactions = find_sig_snp_interactions(snpassoc_bin_results_path, os.path.join(snpassoc_dir, '{}_bin_interactions.csv'))
        
        snpassoc_cont_results_path = os.path.join(snpassoc_dir, '{}_for_snpassoc_cont_snpassoc.csv'.format(group))
        cont_interactions = find_sig_snp_interactions(snpassoc_cont_results_path, os.path.join(snpassoc_dir, '{}_cont_interactions.csv'))
        pd.concat([bin_interactions, cont_interactions]).drop_duplicates().to_csv(os.path.join(snpassoc_dir, '{}_all_interactions.csv'))

        add_snp_interactions(snpassoc_data_path, os.path.join(snpassoc_dir, '{}_all_interactions.csv'), SNPASSOC_FREQ_THRESHOLDS[i])
        
# if __name__ == '__main__':
#     main()
