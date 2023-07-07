from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

import os
import re
import numpy as np
import pandas as pd
import random

class Classifier():
    def __init__(
        self,
        data,
        out_folder
    ):
        '''
        Arguments:
            data -- dataframe containing your data
            out_folder -- path of folder where this instance will output files to
        '''
        self.data=data
        self.out_folder=out_folder

    


    def __split_data(self, target, test_size = 0.25):
        '''
        Wrapper for the sklearn.train_test_split function. Splits self.data to train and test sets

        Arguments:
            target -- target column name (where self.data[target] will be the target array for the sklearn method)
            test_size -- ratio of test dataset size to the total dataset size.
        '''
        X = self.data.drop(columns = [target]).to_numpy()
        y = self.data[target]
        return train_test_split(X, y, stratify = y, test_size = test_size)

    def rdforest(
        self,
        target, 
        n_estimators = 400, 
        max_depth = 7,
        class_weight = 0.5,
        min_samples_split = 2, 
        min_samples_leaf = 1,
        test_size = 0.25
    ):
        '''
        Creates multiple Random Forest classifier models on X_train and y_train with different parameters.
        Runs the models on X_test and compare the results with y_test.

        Args:
            target -- target column identifier on self.data
            n_estimators, max_depth, class_weight, min_samples_split, min_samples_leaf -- hyperparameters 
            test_size -- split ratio for test and training data

        Returns:
            A DataFrame with all the paramaters used and confusion matrices of each model
        '''
        class_weight = {0:class_weight, 1:1-class_weight}
        X_train, X_test, y_train, y_test = self.__split_data(target, test_size)

        rdf = RandomForestClassifier(
            n_estimators = n_estimators, 
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            class_weight = class_weight,
            random_state = 0, 
            n_jobs = -1
        )
        rdf.fit(X_train, y_train)
        
        predicted_scores = rdf.predict_proba(X_test)[:, 1]
        predicted_labels = (predicted_scores > 0.5).astype('int32')

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        f = open(os.path.join(self.out_folder, 'rdf_results.txt'), 'w')
        f.write("Prediction: ")
        unique, counts = np.unique(predicted_labels, return_counts=True)
        f.write(str(dict(zip(unique, counts))))
        f.write(classification_report(y_test, predicted_labels))
        f.close()

        fpr, tpr, _ = roc_curve(y_test, predicted_scores)

        return fpr, tpr, auc(fpr, tpr)


    def xgboost(
        self,
        target,
        learning_rate = 0.05,
        max_depth = 5,
        n_estimators = 300,
        min_child_weight = 3,
        subsample = None,
        colsample_bytree = None,
        scale_pos_weight = 3,
        test_size = 0.25
    ):
        '''
        Creates multiple XgBoost classifier models on X_train and y_train with different parameters.
        Runs the models on X_test and compare the results with y_test.

        Args:
            target -- target column identifier on self.data
            n_estimators, max_depth, class_weight, min_samples_split, min_samples_leaf -- hyperparameters 
            test_size -- split ratio for test and training data

        Returns:
            A DataFrame with all the paramaters used and confusion matrices of each model
        '''
        X_train, X_test, y_train, y_test = self.__split_data(target, test_size)
        
        xgb = XGBClassifier(
            booster = 'gbtree', 
            learning_rate = learning_rate,
            max_depth = max_depth,
            n_estimators = n_estimators,
            min_child_weight = min_child_weight,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            scale_pos_weight = scale_pos_weight,
        )

        xgb.fit(X_train, y_train)
        predicted_scores = xgb.predict_proba(X_test)[:, 1]
        predicted_labels = (predicted_scores > 0.5).astype('int32')

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        f = open(os.path.join(self.out_folder, 'xgb_results.txt'), 'w')
        f.write("Prediction: ")
        unique, counts = np.unique(predicted_labels, return_counts=True)
        f.write(str(dict(zip(unique, counts))))
        f.write(classification_report(y_test, predicted_labels))
        f.close()

        fpr, tpr, _ = roc_curve(y_test, predicted_scores)

        return fpr, tpr, auc(fpr, tpr)


    def naive_bayes(self, X_train, X_test, y_train, y_test):
        '''
        Creates multiple Naive Bayes classifier models on X_train and y_train with different parameters.
        Runs the models on X_test and compare the results with y_test.

        Args:
            X_train: A Numpy array containing the dataset for training
            X_test: A Numpy array containing the dataset for testing
            y_train: A Numpy array consisting of the target values for training
            y_test: A Numpy array consisting of the target values for testing

        Returns:
            A DataFrame with all the paramaters used and confusion matrices of each model
        '''
        bnb = BernoulliNB()
        bnb.fit(X_train, y_train)
        predicted_scores = bnb.predict_proba(X_test)[:, 1]
        predicted_labels = (predicted_scores > 0.5).astype('int32')
        print("Prediction:")
        unique, counts = np.unique(predicted_labels, return_counts=True)
        print(dict(zip(unique, counts)))
        print(classification_report(y_test, predicted_labels))

        fpr, tpr, _ = roc_curve(y_test, predicted_scores)
        return fpr, tpr, auc(fpr, tpr)
        # tn, fp, fn, tp = confusion_matrix(
        #     y_test, predicted_labels, labels=[0, 1]).ravel()
        # convert_matrix_b = [tn, fp, fn, tp]

        # df = df.append({'Confusion Matrix': convert_matrix_b}, ignore_index=True)
        # return df


# Testing the code, doesn't work rn, have to update
def main():

# getting and formatting a test file for log regression
    clinical_df = pd.read_csv("/datalake/AyLab/depression_testing/depression_data_ohc.csv", index_col = 0)
    clinical_df.drop(['Sex', 'Age'], axis=1, inplace=True)
    snp_df = pd.read_csv("/datalake/AyLab/depression_snp_data/FAULTY_dominant_model/dominant_depression_allsnps_6000extra_c1.csv", index_col = 0, nrows = 1000)
    snp_df.drop(['PHQ9_binary', 'Sex'], axis=1, inplace=True) # dropping phq9_binary cuz clinical already has it
    all_cols = snp_df.columns.tolist()
    snp_cols = [col for col in all_cols if col not in ['ID_1']]
    random_10_snps_withID = random.sample(snp_cols, 10) # randomly taking 10 snps for testing
    random_10_snps_withID.append('ID_1') # adding ID to the snps
    snp_df = snp_df[random_10_snps_withID]
    
    snp_and_clinical_df = pd.merge(snp_df, clinical_df, on='ID_1')
    snp_and_clinical_df.dropna(inplace = True) # drop rows with missing values
    snp_and_clinical_ohc_df = snp_and_clinical_df.drop(['Chronotype_1.0', 'Sleeplessness/Insomnia_1.0'], axis=1) # dropping these as they're baseline for OHC

# init values
    binOHCdata = snp_and_clinical_ohc_df.drop('PHQ9', axis=1)
    data = binOHCdata
    out_folder = '/home/akhan/repo_punks_mete/summer23_aylab/data/'
    
# Create an instance of Classifier
    classifier = Classifier(
        data,
        out_folder
    )

    target='PHQ9_binary'
    # # Run rdforest
    classifier.rdforest(target)
    
    # Run xgboost
    classifier.xgboost(target)

    # Run naive_bayes-- not fully implemented yet
    
# Execute the main function
if __name__ == "__main__":
    main()
