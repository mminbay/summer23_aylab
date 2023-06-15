import numpy as np
import pandas as pd
import random
'''
This class is meant to be used to do feature selection after you have compiled your
final dataset. Check an example usage at the end of this file.
'''

class FeatureSelector():
    # TODO: implement init
    def __init__(
        self,
        data,
    ):
        self.data = pd.read_csv(data)

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
        print(ratio)

        target_total = int(ratio * k)
        print(target_total)
        other_total = k - target_total
        print(other_total)
        
        bootstraps = []
        for i in range(0, n):
            this_bootstrap = []
            target_array = list(target_df['ID_1'])
            target_samples = random.sample(target_array, target_total)
            this_bootstrap.append(target_array)
            other_array = list(other_df['ID_1'])
            other_samples = random.sample(other_array, other_total)
            this_bootstrap.append(other_array)
            bootstraps.append(this_bootstrap)

        return bootstraps
    
    def chisquare(X, y, n_features):
        '''
        Runs chi-square feature selection on the data (X) and the target values (y) and finds
        the index of the top n_features number of features.
    
        Args:
            X: A Numpy array containing the dataset
            y: A Numpy array consisting of the target values
            n_features: An integer specifying how many features should be selected
    
        Returns a list containing the indices of the features that have been selected
        '''
        chi2_score, p_val = chi2(X, y)
        index = list(np.argsort(list(chi2_score))[-1*n_features:])
        index.sort()
        return index
    
    def infogain(X, y, n_features):
        '''
        Runs infogain feature selection on the data (X) and the target values (y) and finds
        the index of the top n_features number of features.
    
        Args:
            X: A Numpy array containing the dataset
            y: A Numpy array consisting of the target values
            n_features: An integer specifying how many features should be selected
    
        Returns a list containing the indices of the features that have been selected
        '''
        score = mutual_info_classif(X, y,random_state=0)
        index = list(np.argsort(list(score))[-1*n_features:])
        index.sort()
        return index
    
    def mrmr(X, y, **kwargs):
        """
        This function implements the MRMR feature selection
        Input
        -----
        X: {numpy array}, shape (n_samples, n_features)
            input data, guaranteed to be discrete
        y: {numpy array}, shape (n_samples,)
            input class labels
        kwargs: {dictionary}
            n_selected_features: {int}
                number of features to select
        Output
        ------
        F: {numpy array}, shape (n_features,)
            index of selected features, F[0] is the most important feature
        J_CMI: {numpy array}, shape: (n_features,)
            corresponding objective function value of selected features
        MIfy: {numpy array}, shape: (n_features,)
            corresponding mutual information between selected features and response
        Reference
        ---------
        Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012.
        """
        if 'n_selected_features' in kwargs.keys():
            n_selected_features = kwargs['n_selected_features']
            F, J_CMI, MIfy= LCSI.lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
        else:
            
            F, J_CMI, MIfy = LCSI.lcsi(X, y, gamma=0, function_name='MRMR')
        return F


'''
Below is an example usage
'''

def main():
    featselect = FeatureSelector('/shared/datalake/summer23Ay/hieu_data_imputed_dominant_model_all.csv')
    bootstraps = featselect.bootstrap(10, 100, 'PHQ9_binary', 1)

if __name__ == '__main__':
    main()
            
        
        
    
    