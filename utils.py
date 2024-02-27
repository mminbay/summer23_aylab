import logging
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import re
import requests
from sklearn.model_selection import train_test_split
import subprocess

def capture_snp_identifiers(col, chroms = False):
    '''
    Given a string, return contained expressions that are of the form 'rsXXX' or 'XX_XXXXXX'
    '''
    rs_pattern = r'rs\d+'
    ch_pattern = r'\d{1,2}[_:\-.]\d{6,}_[ATGC]+_[ATGC]+'

    if chroms:
        rs_pattern = r'\d{1,2}_rs\d+'
        ch_pattern = r'\d{1,2}_\d{1,2}[_:\-.]\d{6,}_[ATGC]+_[ATGC]+'
    
    rs_matches = re.findall(rs_pattern, col)
    ch_matches = re.findall(ch_pattern, col)

    matches = []

    if rs_matches and ch_matches:
        if col.index(ch_matches[0]) < col.index(rs_matches[0]):
            matches.extend(ch_matches)
            matches.extend(rs_matches)
        else:
            matches.extend(rs_matches)
            matches.extend(ch_matches)
    elif rs_matches:
        matches.extend(rs_matches)
    elif ch_matches:
        matches.extend(ch_matches)

    if len(matches) > 2:
        raise Exception('{}, length : {}'.format(col, len(matches)))
    return matches

def snpdb_query(
    col,
    max_list = 50,
    assembly = '37', 
    fields = ['chr', 'gene', 'pos'],
    url = 'https://clinicaltables.nlm.nih.gov/api/snps/v3/search'
):
    result = pd.concat([__snpdb_query_helper(snp, max_list, assembly, fields, url) for snp in col], axis = 1)
    result = result.transpose()
    result.rename(columns = {'snp': col.name}, inplace = True)
    return result
 
def __snpdb_query_helper(
    snp, 
    max_list,
    assembly,
    fields,
    url
):
    '''
    Query dbSNP for information on requested SNP.

    Arguments:
        snp (str) -- rsid of snp to be queried
        maxq (int) -- number of results to request
        assembly ('37', '38') -- GRCh assembly to use for queries. Using '37' will also request info on '38' in case it is not available in the former
        fields (list(str)) -- fields to retrieve from dbSNP

    Return:
        result (pandas Series) -- pandas Series containing information on requested fields
    '''
    parameters = ['rsNum']
    parameters.extend(['38.' + field for field in fields])
    if assembly == '37':
        parameters.extend(['37.' + field for field in fields])

    snp_catch = capture_snp_identifiers(snp)
    results = []
    for s in snp_catch:
        if 'rs' in s:
            type = 'rsid'
            terms = s
        else:
            type = 'pos'
            chr = s.split('_')[0]
            pos = s.split('_')[1]
            terms = chr + ', ' + pos
        response = requests.get(
            url,
            params = {
                'terms': terms,
                'maxList': max_list,
                'df': ','.join(parameters),
                'sf': ','.join(parameters)
            }
        ).json() 
        matches = response[3]
        requested_chr_index = parameters.index(assembly + '.chr')
        requested_pos_index = parameters.index(assembly + '.pos')
        found = False
        for match in matches:
            match_rsid = match[0]
            match_chr = match[requested_chr_index]
            match_pos = match[requested_pos_index]
            if type == 'rsid' and match_rsid == s:
                results.append(pd.Series(
                    data = [snp] + match[1:],
                    index = ['snp'] + parameters[1:]
                ))
                found = True
                break
            elif type == 'pos' and match_chr == chr and match_pos == pos:
                results.append(pd.Series(
                    data =  [snp] + match[1:],
                    index = ['snp'] + parameters[1:]
                ))
                found = True
                break
        if not found:
            results.append(pd.Series(
                data = [snp] + ['NA'] * (len(parameters) - 1),
                index = ['snp'] + parameters[1:]
            ))
    if len(results) == 0:
        return pd.Series(
            data = [snp] + ['NA'] * (len(parameters) - 1),
            index = ['snp'] + parameters[1:]
        )
    elif len(results) == 1:
        return results[0]
    elif len(results) == 2:
        result = results[0] + '_' + results[1]
        result['snp'] = snp
        return result
    else:
        raise Exception('something went wrong, query {}, results len {}'.format(snp, len(results)))
    
def filter_snps_from_column(col):
    '''
    Return a boolean array such that an entry is true if the corresponding variable in the argument column started with 'rs' or 'pair' or 'SNP'.
    '''
    return col.str.startswith('SNP') | col.str.startswith('pair') | col.str.startswith('rs')

def split_pair(pair, delim = '-'):
    '''
    Split a pair and return a tuple with two strings. If provided string is not a pair, return the string directly.

    Arguments:
        pair (str) -- string to split. it should start with 'pair:' or 'pair_'
    '''
    if 'pair' not in pair:
        return pair
    return pair[5:].split(delim)[:2]

def find_sig_snp_interactions(csv_file, output_file = None):
    """
    Find significant positions in a symmetric matrix CSV file.
    Given a CSV file containing a symmetric matrix, this function identifies all positions
    (row-name, column-name) in the upper triangular part (excluding the diagonal) where the
    values are less than 0.05. This is made to work well with the csv output of snpassoc since 
    upper triangular of that output file has epistatic snp-snp interactions needed for
    network analysis.
    Parameters:
        csv_file (str): The file path of the CSV file containing the symmetric matrix.
        output_file (str, optional): The file path where the DataFrame will be saved as CSV.
    Returns:
        pandas DataFrame: A two-column DataFrame, where each row has two SNPs with an edge.
        Structure is fit to be used in network analysis.
    Example:
        csv_file_path = 'path/to/your/csv_file.csv'
        result = find_sig_snp_interactions(csv_file_path)
        print(result)
        # Optionally, save the DataFrame to a CSV file
        output_csv_path = 'path/to/output.csv'
        find_sig_snp_interactions(csv_file_path, output_file=output_csv_path)
    """
    # Rest of the function implementation remains the same
    df = pd.read_csv(csv_file, index_col=0)
    significant_positions = set()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            value = df.iloc[i, j]
            if value < 0.05:
                significant_positions.add((df.index[i], df.columns[j]))

    result_df = pd.DataFrame(list(significant_positions), columns=['SNP1', 'SNP2'])

    if output_file:
        result_df.to_csv(output_file, index=False)

    return result_df

def snpassoc_wrapper(script_path, data_path, outcome_var, outcome_type):
    # Construct the command to run the R script with arguments
    command = ["Rscript", script_path, data_path, outcome_var, outcome_type]
    
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
        print(process.stdout)
        print(process.stderr)

def reorder_cols(data, cols = ['ID_1', 'PHQ9', 'PHQ9_binary']):
    '''
    Return a view of a dataframe where the provided columns are at the end.

    Arguments:
        data (DataFrame) -- dataframe to be reordered.
        cols (str, or list(str)) -- columns to be pushed to the end.
    '''
    if type(cols) == str:
        cols = [cols]
    new_cols_order = [col for col in data.columns if col not in cols] + [col for col in cols if col in data.columns]
    return data[new_cols_order]

def merge_for_fs(snp_data, depression_path, sex = None, outcome = ['PHQ9_binary']):
    '''
    Create a dataframe that is ready for feature selection. This dataframe will be filtered for sex if required,
    and will only have SNP columns, ID_1, and the outcome variable columns.

    Arguments:
        snp_data (DataFrame) -- dataframe containing SNP information and ID_1. there should be no other columns
        depression_path (str) -- path to .csv file containing clinical factor and PHQ9 data
        sex ('male' or 'female') -- if provided, result dataframe will be filtered for sex
        outcome (str, or list(str)) -- (list of) identifier(s) of the outcome variable column.
    '''
    if type(outcome) == str:
        outcome = [outcome]
    depression_data = pd.read_csv(depression_path, usecols = ['ID_1', 'Sex'] + outcome)
    final = snp_data.merge(depression_data, on = 'ID_1')
    if sex == 'male':
        final = final[final['Sex'] == 1]
    elif sex == 'female':
        final = final[final['Sex'] == 0]
    # if outcome == 'cont':
    #     final.drop(columns = ['PHQ9_binary'], inplace = True)
    # elif outcome == 'bin':        
    #     final.drop(columns = ['PHQ9'], inplace = True)
    # elif outcome == 'both':
    #     pass
    # else:
    #     raise Exception('\'outcome\' parameter must be \'bin\' or \'cont\' or \'both\'')
    return final.drop(columns  = ['Sex'])

def compile_fs_results(dir, identifier = None, out_name = 'compiled.csv'):
    '''
    Compile feature selection results from different folders in a directory. This function assumes that you have created multiple FeatureSelector object for feature selection, which created their separate directories in a common directory. It is also assumed that no folder in this directory is shared by two FS objects, as in, a single folder will only contain a single summary .csv file and a single bootstraps .csv file, and all the summary files have the same columns with no duplicates. Outputs the results at the given directory with the provided output name

    Arguments:
        dir (str) -- path to directory to look for feature selector directories at.
        identifier (str) -- only look at directories whose name contains this substring. useful if you outputted multiple feature selection directories (e.g. chi2 and t-test) to the same parent directory.
        out_name (str) -- name of compiled output file
    '''
    if identifier is None:
        folder_paths = [os.path.join(dir, folder) for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder)) and 'ipynb' not in folder]
    else:
        folder_paths = [os.path.join(dir, folder) for folder in os.listdir(dir) if os.path.isdir(os.path.join(dir, folder)) and identifier in folder and 'ipynb' not in folder]

    dfs = []
    for path in folder_paths:
        file = [file for file in os.listdir(path) if '.csv' in file and 'bootstraps' not in file][0]
        df = pd.read_csv(os.path.join(path, file), index_col = 0)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(os.path.join(dir, out_name))
    
def add_snp_interactions(data_path, interactions_path, freq_threshold, verbose = False):
    '''
    Add SNP interactions as columns to a dataframe. A SNP interaction pair:rs1-rs2 is true for a sample if rs1 is true and rs2 is true for that sample. Interactions that are true for for less than freq_threshold samples will not be considered.

    Arguments:
        data_path (str) -- path to .csv dataframe to add the columns to.
        interactions_path (str) -- path to the .csv file containing the interactions. this file should have SNP1 and SNP2 columns, where every row indicates an interaction. this file is usually outputted by preprocessing of network analysis.
        freq_threshold (int) -- interactions that occur less than this number will not be included in the table
        verbose (bool) -- if True, output a separate file that contains frequency information about all interactions
    '''
    data = pd.read_csv(data_path)
    interactions = pd.read_csv(interactions_path)

    for i in range(len(interactions)):
        snp1 = interactions.iloc[i]['SNP1']
        snp2 = interactions.iloc[i]['SNP2']

        interaction_column = (data[snp1] & data[snp2]).astype(int)
        if verbose:
            print(snp1, snp2, interaction_column.sum())
        if interaction_column.sum() < freq_threshold:
            continue
        data['pair:' + snp1 + '-' + snp2] = interaction_column

    data.to_csv(data_path.split('.')[0] + '_wpairs.csv', index = False)
    
def compile_snps(snps, dir, out_path, factors = ['ID_1']):
    '''
    Compile a dataframe of given list of SNPs across all chrom files. Output as .csv.

    Arguments:
        snps (set(str)) -- set of snps to be compiled.
        dir (str) -- path to directory of chrom files. this directory should only contain .csv files of the chroms
        out_path (str) -- path where the result will be outputted
        factors (list(str)) -- LIST of fix columns across all chrom files (almost deprecated)
    '''
    logging.basicConfig(filename = os.path.join(os.path.dirname(out_path), 'compiler.log'), encoding='utf-8', level=logging.DEBUG)
    files = [file for file in os.listdir(dir) if '.csv' in file]
    individual_args = [(os.path.join(dir, file), snps, factors) for file in files]
    
    with Pool() as pool:
        results = pool.map(compile_wrapper, individual_args)

    final = results[0]
    for df in results[1:]:
        final = final.merge(df, on = 'ID_1')
    final = final.loc[:,~final.columns.duplicated()].copy()
    final.to_csv(out_path)
        
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
    result = pd.concat(result, axis = 1)
    print(len(result))
    return result

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