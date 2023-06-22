import numpy as np
import pandas as pd
import os
import random
from skfeature.function.information_theoretical_based import LCSI
from sklearn.feature_selection import chi2
from statsmodels.stats.multitest import fdrcorrection
from sklearn.feature_selection import mutual_info_classif
'''
This class is meant to be used to do feature selection after you have compiled your
final dataset. Check an example usage at the end of this file.
'''

class FeatureSelector():
    # TODO: implement init
    def __init__(
        self,
        data,
        drop,
        filter,
        out_folder
    ):
        '''
        Arguments:
        data -- path to .csv file to run feature selection on
        drop -- columns to drop from data. the resultant dataframe should only have the columns to run feature selection on
        filter -- list of tuples (target column: target value) that will be used to filter the dataset. useful for filtering for sex
        out_folder -- where this instance will write files to
        '''
        self.data = pd.read_csv(data)
        self.data.drop(columns = drop, inplace = True)
        self.out_folder = out_folder

    def bootstrap(self, n, k, target_column, target_outcome):
        '''
        Return a n-size list of k-size stratified random samples with replacements.

        Arguments:
        n -- number of bootstraps
        k -- number of samples in of each bootstrap
        target_column -- column to check for stratifying
        target_outcome -- value to check for in target_column

        Returns:
        bootstraps -- nested list of indices for every bootstrap
        '''

        target_df = self.data[self.data[target_column] == target_outcome]
        other_df = self.data[self.data[target_column] != target_outcome]
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
            
    
    def chisquare(self, data, target, out_name):
        '''
        Runs chi-square feature selection on the data and the target values. Outputs
        results to a .csv at given path
    
        Arguments:
            data -- dataset (in DataFrame type)
            target -- target column label in dataset
            outname -- name of the file to which the results will be outputted
        '''
        target_arr = data[target].to_numpy().astype('int')
        only_snp_data = data.drop(columns = [target, 'ID_1'])
        data_arr = only_snp_data.to_numpy()
        
        chi2_score, p_val = chi2(data_arr, target_arr)
        df = pd.DataFrame()
        df["SNP"] = only_snp_data.columns
        df["chi2_score"] = chi2_score.tolist()
        df["p_val"] = p_val.tolist()
        # TODO: normalized score!
        df['normalized_score'] = np.arange(0, len(df))
        df.sort_values(by="chi2_score", inplace=True, ascending = False)

        df.to_csv(out_name)
    
    def infogain(self, data, target, out_name):
        '''
        Runs infogain feature selection on the data and the target values. Outputs
        results to a .csv at given path
    
        Arguments:
            data -- dataset (in DataFrame type)
            target -- target column label in dataset
            outname -- name of the file to which the results will be outputted
        '''
        target_arr = data[target].to_numpy().astype('int')
        only_snp_data = data.drop(columns = [target, 'ID_1'])
        data_arr = only_snp_data.to_numpy()
        score = mutual_info_classif(data_arr, target_arr, random_state=0)
        df = pd.DataFrame()
        df["SNP"] = only_snp_data.columns
        df["infogain_score"] = score.tolist()
        # TODO: normalized score!
        df['normalized_score'] = np.arange(0, len(df))
        df.sort_values(by="infogain_score", inplace=True, ascending = False)

        df.to_csv(out_name)
    
    def mrmr(self, data, target, out_name, **kwargs):
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
        # TODO: normalized score!
        df['normalized_score'] = np.arange(0, len(df))
        df.to_csv(out_name)
        
    def jmi(self, data, target, out_name, **kwargs):
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
        # TODO: normalized score!
        df['normalized_score'] = np.arange(0, len(df))
        df.to_csv(out_name)

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
            if not os.path.exists(selector_folder):
                os.makedirs(selector_folder)
            for j in range(int(each_selector)):
                index = i * each_selector + j
                sample = self.get_sample(bootstraps[index])
                filename = os.path.join(selector_folder, selector_names[i] + '_' + str(index) + '.csv')
                function(sample, target_column, filename)
                filenames.append(filename)

        final = pd.DataFrame()
        snps = pd.read_csv(filenames[0])['SNP']
        final['SNP'] = snps
        final['total_normalized_score'] = np.zeros(len(snps))
        for file in filenames:
            curr_file = pd.read_csv(file)
            for snp in final['SNP']:
                final.loc[final['SNP'] == snp, 'total_normalized_score'] += curr_file.loc[final['SNP'] == snp, 'normalized_score']

        final.sort_values(by = 'total_normalized_score', ascending = False)
        final.to_csv(os.path.join(self.out_folder, out_name + '.csv'))

'''
Below is an example usage
'''

def main():
    drop = [
        'Sex',
        'Age',
        'Unnamed: 0',
        'Unnamed: 0.1',
        'Unnamed: 0.2',
        'Lifetime number of depressed periods',
        'Bipolar and major depression status',
        'PHQ9',
        'PHQ9_multiclass',
        'Chronotype',
        'Getting up in the morning',
        'Prolonged feeling of sadness/depression',
        'Sleeplessness/Insomnia'   
    ]
    fselect = FeatureSelector(
        '/home/mminbay/summer_research/summer23_aylab/data/kremaliborek.csv', 
        drop, 
        [], 
        '/home/mminbay/summer_research/summer23_aylab/data/feat_select/'
    )
    fselect.bootstrapped_feat_select(10, 1000, 'PHQ9_binary', 1, [fselect.chisquare, fselect.infogain], ['chi2', 'infogain'], 'test')

    
    # bootstraps = fselect.bootstrap(10, 500, 'PHQ9_binary', 1)

    # name = '{}_{}'
    # for i in range (0, 5):
    #     method = 'chi2'
    #     bootstrap = bootstraps[i]
    #     sample = fselect.get_sample(bootstrap)
    #     fselect.chisquare(sample, 'PHQ9_binary', name.format(method, str(i + 1))+'.csv')

    # for i in range (5, 10):
    #     method = 'infogain'
    #     bootstrap = bootstraps[i]
    #     sample = fselect.get_sample(bootstrap)
    #     fselect.infogain(sample, 'PHQ9_binary', name.format(method, str(i + 1))+'.csv')
    

if __name__ == '__main__':
    main()
            
        
        
    
    