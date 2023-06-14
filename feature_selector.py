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

'''
Below is an example usage
'''

def main():
    featselect = FeatureSelector('/shared/datalake/summer23Ay/hieu_data_imputed_dominant_model_all.csv')
    bootstraps = featselect.bootstrap(10, 100, 'PHQ9_binary', 1)

if __name__ == '__main__':
    main()
            
        
        
    
    