import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from multiprocessing import pool
import subprocess

def snpassoc_wrapper(script_path, data_path, outcome):
    # Construct the command to run the R script with arguments
    command = ["Rscript", script_path, data_path, outcome]
    
    # Execute the R script through the command line
    process = subprocess.run(command, capture_output=True, text=True)
    
    # Check the return code and print the output and errors (if any)
    if process.returncode == 0:
        print("R script executed successfully.")
        print("Output:")
        print(process.stdout)
    else:
        print("Error occurred while running the R script.")
        print("Errors:")
        print(process.stderr)

def reorder_cols(data, cols = ['ID_1', 'PHQ9', 'PHQ9_binary']):
    '''
    Return a view of a dataframe where the provided columns are at the end. 
    '''
    new_cols_order = [col for col in data.columns if col not in cols] + cols
    return data[new_cols_order]

def merge_for_fs(snp_data, depression_path, sex = None, outcome = 'bin'):
    '''
    Create a dataframe that is ready for feature selection. This dataframe will be filtered for sex if required,
    and will only have SNP columns, ID_1, and PHQ9_binary or PHQ9 as columns

    Arguments:
        snp_data (DataFrame) -- dataframe containing SNP information and ID_1. there should be no other columns
    '''
    depression_data = pd.read_csv(depression_path, index_col = 0)
    final = snp_data.merge(depression_data, on = 'ID_1')
    if sex == 'male':
        final = final[final['Sex'] == 1]
    elif sex == 'female':
        final = final[final['Sex'] == 0]
    if outcome == 'cont':
        final.drop(columns = ['PHQ9_binary'], inplace = True)
    elif outcome == 'bin':        
        final.drop(columns = ['PHQ9'], inplace = True)
    else:
        raise Exception('\'outcome\' parameter must be \'bin\' or \'cont\'')
    return final

def compile_fs_results(dir, identifier = None, out_name = 'compiled.csv'):
    '''
    Compile feature selection results from different folders in a directory. This function assumes that you have created multiple FeatureSelector object for feature selection, which created their separate directories in a common directory. It is also assumed that no folder in this directory is shared by two FS objects, as in, a single folder will only contain a single summary .csv file and a single bootstraps .csv file, and all the summary files have the same columns with no duplicates. Outputs the results at the given directory with the provided output name

    Arguments:
        dir (str) -- path to directory to look for feature selector directories at.
        identifier (str) -- only look at directories whose name contains this substring. useful if you outputted multiple feature selection directories (e.g. chi2 and t-test) to the same parent directory.
        output_name (str) -- name of compiled output file
    '''
    if identifier is None:
        folder_paths = [os.path.join(dir, folder) for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder))]
    else:
        folder_paths = [os.path.join(dir, folder) for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder)) and identifier in folder]

    dfs = []
    for path in folder_paths:
        file = [file for file in os.listdir(path) if '.csv' in file and 'bootstraps' not in file][0]
        df = pd.read_csv(os.path.join(path, file), index_col = 0)
        files.append(df)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(dir, output_name))
    
def add_snp_interactions(data_path, interactions_path, freq_threshold, verbose = False):
    '''
    Add SNP interactions as columns to a dataframe. A SNP interaction pair:rs1-rs2 is true for a sample if rs1 is true and rs2 is true for that sample. Interactions that are true for for less than freq_threshold samples will not be considered.

    Arguments:
        data_path (str) -- path to .csv dataframe to add the columns to.
        interactions_path (str) -- path to the .csv file containing the interactions. this file should have SNP1 and SNP2 columns, where every row indicates an interaction. this file is usually outputted by preprocessing of network analysis.
        freq_threshold (int) -- interactions that occur less than this number will not be included in the table
        verbose (bool) -- if True, output a separate file that contains frequency information about all interactions
    '''
    data = pd.read_csv(data_path, index_col = 0)
    interactions = pd.read_csv(interactions_path)

    for i in range(len(interactions)):
        snp1 = interactions.iloc[i]['SNP1']
        snp2 = interactions.iloc[i]['SNP2']

        interaction_column = (data[snp1] & data[snp2]).astype(int)
        print(snp1, snp2, interaction_column.sum())
        if interaction_column.sum() < freq_threshold:
            continue
        data['pair:' + snp1 + '-' + snp2] = interaction_column

    data.to_csv(data_path.split('.')[0] + '_wpairs.csv')
    
def compile_snps(snps, factors, dir, out):
    '''
    Compile a dataframe of given list of SNPs across all chrom files. Output as .csv.

    Arguments:
        snps (set(str)) -- set of snps to be compiled.
        factors (list(str)) -- LIST of fix columns (Sex, ID_1, PHQ9_binary)
        dir (str) -- path to directory of chrom files. this directory should only contain .csv files of the chroms
        out_file (str) -- path to folder to output the result
    '''
    logging.basicConfig(filename= os.path.join(os.path.dirname(out), 'compiler.log'), encoding='utf-8', level=logging.DEBUG)
    files = [file for file in os.listdir(dir) if '.csv' in file]
    individual_args = [(os.path.join(dir, file), snps, factors) for file in files]
    
    with Pool() as pool:
        results = pool.map(compile_wrapper, individual_args)

    final = pd.concat(results, axis = 1)
    final = final.loc[:,~final.columns.duplicated()].copy()

    final.to_csv(out)
        
def compile_wrapper(args):
    '''
    DO NOT CALL DIRECTLY
    Used for parallelization in compiling a list of SNPs from all chrom files.

    Arguments:
        args (tuple) -- a tuple that should contain two fields
            args[0] (str) -- path to chrom .csv file
            args[1] (set(str)) -- set of SNP columns to be compiled
            args[2] (list(str)) -- list of fix columns (Sex, ID_1, PHQ9_binary)
    '''
    logging.info('pid: {}. Working on {}.'.format(os.getpid(), args[0]))
    df = pd.read_csv(args[0], index_col = 0)
    snp_set = set(df.columns.tolist())
    snps_present = args[1].intersection(snp_set)
    logging.info('pid: {}. Found {} SNPs on {}.'.format(os.getpid(), str(len(snps_present)), args[0]))
    df.sort_values('ID_1', inplace = True)
    result = []
    sum_na = 0
    for snp in snps_present:
        this_na = df[snp].isna().sum()
        sum_na += this_na
        logging.info('{} has {} NaN values'.format(snp, str(this_na)))
        result.append(df[snp])
    for col in args[2]:
        result.append(df[col])
    logging.info('{} has {} NaN in total out of {}'.format(os.path.basename(args[0]), str(sum_na), len(df.index)))
    return pd.concat(result, axis = 1)

def train_test(path, column, test_size):
    '''
    Split given dataset to train and test sets, stratified. Output both as .csv

    Arguments:
        path (str) -- path to .csv file with the pandas dataframe
        column (str) -- column label in the dataframe to stratify the data accordingly
        test_size (float) -- test set size, between 0 and 1.

    Outputs results in the same directory, with the same file name as the input but with '_train' and '_test' before '.csv' extension
    '''
    df = pd.read_csv(path, index_col = 0)
    X = df.drop(columns = [column])
    y = df[column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y)
    X_train[column] = y_train
    X_test[column] = y_test
    
    train_path = os.path.splitext(path)[0] + '_train.csv'
    test_path =os.path.splitext(path)[0] + '_test.csv'
    
    X_train.to_csv(train_path)
    X_test.to_csv(test_path)