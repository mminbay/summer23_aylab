import numpy as np
import pandas as pd
import os
import random
import time
import logging
from skfeature.function.information_theoretical_based import LCSI
from sklearn.feature_selection import chi2
from statsmodels.stats.multitest import fdrcorrection
from sklearn.feature_selection import mutual_info_classif
import multiprocessing as mp
from multiprocessing import Pool, Value, Manager
'''
This class is meant to be used to do feature selection after you have compiled your
final dataset. Check an example usage at the end of this file.
'''

def fs_wrapper(args):
    '''
    Used for parallelization.

    Arguments:
        args -- a tuple that should contain two fields
            args[0] -- rsid of snp
            args[1] -- namespace that contains three fields
                ns.data -- reference to the dataframe with predictors (snps) and target 
                ns.target_arr -- target array
                ns.func -- feature selection function that will be used
    '''
    predictor = np.reshape(args[1].data[args[0]].to_numpy(), (-1, 1))
    target = args[1].target_arr
    fs_func = args[1].func
    result = fs_func(predictor, target)
    return result, args[0]

def chisquare(data, target, out_name):
    '''
    Runs parallelized chi-square feature selection on the data and the target values. Outputs
    results to a .csv at given path

    Arguments:
        data -- dataset (in DataFrame type)
        target -- target column label in dataset
        outname -- name of the file to which the results will be outputted
    '''
    start_time = time.time()
    # logging.info('Started rounds of feature selection at: ' + str(start_time))
    target_arr = data[target].to_numpy().astype('int')
    only_snp_data = data.drop(columns = [target, 'ID_1'])

    mgr = Manager() # to share the same dataframe across multiple instances
    ns = mgr.Namespace()
    ns.data = only_snp_data
    ns.func = chi2
    ns.target_arr = target_arr
    
    individual_args = [(snp, ns) for snp in only_snp_data]

    parallel_time = time.time()
    # logging.info('Started parallelization of feature selection at: ' + str(parallel_time))
    # logging.info('Overhead to start parallelizing: ' + (str(parallel_time - start_time)))
    with Pool() as pool:
        results = pool.map(fs_wrapper, individual_args)
    mgr.shutdown()
    
    end_time = time.time()
    # logging.info('Stopped parallelization of feature selection at: ' + str(parallel_time))
    # logging.info('Parallelized step took: ' + (str(end_time - parallel_time)))

    df = pd.DataFrame()
    df["SNP"] = [result[1] for result in results]
    df["chi2_score"] = [result[0][0][0] for result in results]
    df["p_val"] = [result[0][1][0] for result in results]
    df.sort_values(by="chi2_score", inplace=True, ascending = False)
    df['rank'] = np.arange(0, len(df))

    df.to_csv(out_name)

def infogain(data, target, out_name):
    '''
    Runs infogain feature selection on the data and the target values. Outputs
    results to a .csv at given path

    Arguments:
        data -- dataset (in DataFrame type)
        target -- target column label in dataset
        outname -- name of the file to which the results will be outputted
    '''
    start_time = time.time()
    # logging.info('Started rounds of feature selection at: ' + str(start_time))
    target_arr = data[target].to_numpy().astype('int')
    only_snp_data = data.drop(columns = [target, 'ID_1'])

    mgr = Manager() # to share the same dataframe across multiple instances
    ns = mgr.Namespace()
    ns.data = only_snp_data
    ns.func = mutual_info_classif
    ns.target_arr = target_arr
    
    individual_args = [(snp, ns) for snp in only_snp_data]

    parallel_time = time.time()
    # logging.info('Started parallelization of feature selection at: ' + str(parallel_time))
    # logging.info('Overhead to start parallelizing: ' + (str(parallel_time - start_time)))
    with Pool() as pool:
        results = pool.map(fs_wrapper, individual_args)

    mgr.shutdown()

    end_time = time.time()
    # logging.info('Stopped parallelization of feature selection at: ' + str(parallel_time))
    # logging.info('Parallelized step took: ' + (str(end_time - parallel_time)))
    
    df = pd.DataFrame()
    df["SNP"] = [result[1] for result in results]
    df["infogain_score"] = [result[0] for result in results]
    df.sort_values(by="infogain_score", inplace=True, ascending = False)
    df['rank'] = np.arange(0, len(df))
    
    df.to_csv(out_name)

def mrmr(data, target, out_name, **kwargs):
    '''
    Applies MRMR feature selection on the data and target values. Outputs
    results to a .csv at given path

    Arguments:
        data -- dataset (in DataFrame type)
        target -- target column label in dataset
        outname -- name of the file to which the results will be outputted
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    '''
    target_arr = data[target].to_numpy().astype('int')
    only_snp_data = data.drop(columns = [target, 'ID_1'])
    data_arr = only_snp_data.to_numpy()
    
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy= LCSI.lcsi(data_arr, target_arr, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
    else:   
        F, J_CMI, MIfy = LCSI.lcsi(data_arr, target_arr, gamma=0, function_name='MRMR')
        
    df = pd.DataFrame()
    chosen_snps = []
    for index in F:
        chosen_snps.append(list(only_snp_data.columns)[index])
    df["SNP"] = chosen_snps
    df['rank'] = np.ones(len(df))
    df.to_csv(out_name)
    
def jmi(data, target, out_name, **kwargs):
    '''
    Applies MRMR feature selection on the data and target values. Outputs
    results to a .csv at given path

    Arguments:
        data -- dataset (in DataFrame type)
        target -- target column label in dataset
        outname -- name of the file to which the results will be outputted
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
    '''
    target_arr = data[target].to_numpy().astype('int')
    only_snp_data = data.drop(columns = [target, 'ID_1'])
    data_arr = only_snp_data.to_numpy()
    
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = LCSI.lcsi(data_arr, target_arr, function_name='JMI', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = LCSI.lcsi(data_arr, target_arr, function_name='JMI')
    return

    df = pd.DataFrame()
    chosen_snps = []
    for index in F:
        chosen_snps.append(list(only_snp_data.columns)[index])
    df["SNP"] = chosen_snps
    df['rank'] = np.ones(len(df))
    df.to_csv(out_name)

class FeatureSelector():
    # TODO: implement init
    def __init__(
        self,
        data,
        out_folder
    ):
        '''
        Arguments:
            data -- dataset (DataFrame) that contains features and target. should only contain features that will undergo feature selection, the ID_1 column, and the target column
            out_folder -- where this instance will write files to
        '''
        self.data = data
        self.out_folder = out_folder

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        logging.basicConfig(filename= os.path.join(out_folder, 'feature_selector.log'), encoding='utf-8', level=logging.DEBUG)

    def bootstrap(self, n, k, target_column, target_outcome):
        '''
        Return a n-size list of k-size stratified random samples with replacements.

        Arguments:
            n -- number of bootstraps
            k -- number of samples in of each bootstrap. pass -1 for the bootstrap size to be the same as the total sample sizes
            target_column -- column to check for stratifying
            target_outcome -- value to check for in target_column

        Returns:
            bootstraps -- nested list of ID_1's for every bootstrap
        '''

        target_df = self.data[self.data[target_column] == target_outcome]
        other_df = self.data[self.data[target_column] != target_outcome]

        if k == -1:
            target_total = len(target_df.index)
            other_total = len(other_df.index)
        else:
            ratio = len(target_df.index) / len(self.data.index)
            target_total = int(ratio * k)
            other_total = k - target_total
        
        bootstraps = []
        for i in range(0, n):
            this_bootstrap = []
            target_array = list(target_df['ID_1'])
            target_samples = random.sample(target_array, target_total)
            this_bootstrap.extend(target_samples)
            other_array = list(other_df['ID_1'])
            other_samples = random.sample(other_array, other_total)
            this_bootstrap.extend(other_samples)
            bootstraps.append(this_bootstrap)

        return bootstraps

    def load_bootstraps(self, file):
        '''
        Read a list of bootstrap participant ID's from given dataframe, and return it.
        Useful for maintaining the same sample across different runs.

        Arguments:
            file -- path to the .csv file containing the bootstraps, where every column
                should be a list of ID_1's for a bootstrap.

        Returns:
            bootstraps -- nested list of ID_1's for every bootstrap
        '''
        bootstraps = []
        df = pd.read_csv(file, index_col = 0)
        for col in df.columns:
            bootstraps.append(df[col].tolist())
        return bootstraps

    def get_sample(self, ids):
        '''
        Return a dataframe containing the participants with given list of ids

        Arguments:
            ids -- list of ID_1's (can have repetitions)

        Returns:
            sample -- slice from self.data containing participants with given ids
        '''
        result = pd.DataFrame(columns = self.data.columns)
        for id in ids:
            result = pd.concat([result, self.data[self.data['ID_1'] == id]])
        return result

    def bootstrapped_feat_select(self, n, k, target_column, target_outcome, selectors, selector_names, out_name):
        '''
        Bootstraps the dataset, applies feature selection with specified functions.
        Compiles results across all and outputs them as a .csv file.

        Arguments:
            n -- number of bootstraps
            k -- number of samples in each bootstrap
            target_column -- column to check for stratifying
            target_outcome -- value to check for in target_column
            selectors -- list of functions to use for feature selection. n % len(selectors) should be 0,
                and every selector will be applied equal times. you can repeat selectors in list to manipulate this
            selector_names -- list of function names that will be used to name subdirectories.
            out_name -- name of subdirectories and compiled results file that will be created
        '''
        selector_len = len(selectors)
        if n % len(selectors) != 0:
            raise Exception('Selectors must evenly divide the number of bootstraps')
            
        if selector_len != len(selector_names):
            raise Exception('Selector list must be as long as selector name list')

        bootstraps = self.bootstrap(n, k, target_column, target_outcome)
        each_selector = int(n / selector_len)
        filenames = []
        
        for i in range(selector_len):
            function = selectors[i]
            selector_folder = os.path.join(self.out_folder, out_name + '_' + selector_names[i])
            logging.info(selector_folder)
            if not os.path.exists(selector_folder):
                os.makedirs(selector_folder)
                logging.info('made a folder ' + selector_folder)
            for j in range(int(each_selector)):
                index = i * each_selector + j
                sample = self.get_sample(bootstraps[index])
                filename = os.path.join(selector_folder, selector_names[i] + '_' + str(index) + '.csv')
                function(sample, target_column, filename)
                filenames.append(filename)

        final = pd.DataFrame()
        snps = pd.read_csv(filenames[0])['SNP']
        final['SNP'] = snps
        final['total_rank'] = np.zeros(len(snps))
        for file in filenames:
            curr_file = pd.read_csv(file)
            for snp in final['SNP']:
                final.loc[final['SNP'] == snp, 'total_rank'] += curr_file.loc[final['SNP'] == snp, 'rank']

        final.sort_values(by = 'total_rank', ascending = False)
        final.to_csv(os.path.join(self.out_folder, out_name + '.csv'))

        bootstrap_df = pd.DataFrame()
        for i in range(len(bootstraps)):
            name = 'bootstrap_' + str(i + 1)
            bootstrap_df[name] = bootstraps[i]

        bootstrap_df.to_csv(os.path.join(self.out_folder, out_name + '_bootstraps.csv'))
            

'''
Below is an example usage
'''

'''
def main():
    data = pd.read_csv('/home/mminbay/summer_research/summer23_aylab/data/imputed_data/final_data/final_depression_allsnps_6000extra_c1.csv', index_col = 0)

    data.drop(columns = ['Sex'], inplace = True)
    
    fselect = FeatureSelector(
        data, 
        '/home/mminbay/summer_research/summer23_aylab/data/feat_select/'
    )
    fselect.bootstrapped_feat_select(10, 1000, 'PHQ9_binary', 1, [fselect.chisquare, fselect.infogain], ['chi2', 'infogain'], 'test')
    
if __name__ == '__main__':
    main()
'''
            
        
        
    
    