import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from imblearn import FunctionSampler
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, balanced_accuracy_score, fbeta_score
from skopt import BayesSearchCV
from sklearn.linear_model import  SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.discriminant_analysis as sk
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skopt.space import Real, Integer, Categorical
import multiprocessing as mp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from imblearn.combine import SMOTETomek
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import re
import os
from ast import literal_eval

import logging

'''
This file is meant to be used for classification. An example usage can be found at the end of this file.
'''

class MyClassifier(BaseEstimator, ClassifierMixin):
    '''
    Classifier wrapper. DO NOT CALL THIS DIRECTLY
    '''
    def __init__(
        self, estimator, method, path, splits, run, n_estimators = 100, max_depth = 10, 
        penalty = 'none', alpha = 0.01, n_neighbors = 5, solver = 'svd', shrinkage = 0,
        kernel = 'linear', C = 1, gamma = 0.1, degree = 3, eta = 0.05, scale_pos_weight = 1,
        min_samples_split = 2, min_samples_leaf = 1, l1_ratio = 0.15, leaf_size = 30,
        subsample = 3, colsample_bytree = 5, min_child_weight = 3, class_weight = 0.5,
        hidden_layer_sizes = (64, 32, 16, 8), fbeta = None
    ):

        self.idx, self.method, self.estimator, self.splits = 0, method, estimator, splits
        self.max_depth, self.n_estimators = max_depth, n_estimators
        self.penalty, self.alpha, self.n_neighbors = penalty, alpha, n_neighbors
        self.solver, self.shrinkage = solver, shrinkage
        self.C, self.gamma, self.degree, self.kernel = C, gamma, degree, kernel
        self.eta, self.repeat, self.path = eta, -1, path
        self.scale_pos_weight, self.min_child_weight = scale_pos_weight, min_child_weight
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.l1_ratio, self.leaf_size, self.class_weight = l1_ratio, leaf_size, class_weight
        self.subsample, self.colsample_bytree = subsample, colsample_bytree,
        self.run = run
        self.hidden_layer_sizes = hidden_layer_sizes
        self.fbeta = fbeta


        if self.estimator == 'rdforest':
            self.classify = RandomForestClassifier(n_estimators=self.n_estimators, max_depth= self.max_depth,
            min_samples_split=self.min_samples_split, min_samples_leaf = self.min_samples_leaf)
            self.hyperParam = ['class_weight', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
            
        elif self.estimator == 'logreg':
            self.classify = LogisticRegression(penalty = self.penalty, max_iter=10000)
            self.hyperParam = ['class_weight', 'penalty']
        
        elif self.estimator == 'elasticnet':
            self.classify = SGDClassifier(loss = 'log_loss', alpha= self.alpha, penalty = 'l1', 
            random_state=0, l1_ratio = self.l1_ratio)
            self.hyperParam = ['alpha', 'class_weight', 'l1_ratio']
        
        elif self.estimator == 'naiveBayes':
            self.classify = BernoulliNB(alpha=self.alpha)
            self.hyperParam = ['alpha']

        elif self.estimator == 'knn':
            self.classify = KNeighborsClassifier(n_neighbors=self.n_neighbors, leaf_size = self.leaf_size)
            self.hyperParam = ['leaf_size', 'n_neighbors']

        elif self.estimator == 'LDA':
            self.classify = sk.LinearDiscriminantAnalysis(solver = self.solver, shrinkage=self.shrinkage)
            self.hyperParam = ['shrinkage', 'solver']
        
        elif self.estimator == 'svm':
            # if self.kernel == 'linear':
            #     self.classify = SVC(kernel = 'linear', C=self.C, random_state=0, max_iter=100000)
            # elif self.kernel == 'rbf':
            #     self.classify = SVC(kernel = 'rbf', C=self.C, gamma=self.gamma, random_state=0, max_iter=100000)
            # elif self.kernel == 'poly':
            self.classify = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, random_state=0, max_iter=10000000)
            self.hyperParam = ['C', 'class_weight', 'degree', 'gamma', 'kernel']

        elif self.estimator == 'xgboost':
            self.classify = XGBClassifier(booster='gbtree',max_depth=self.max_depth,eta=self.eta,
            min_child_weight = self.min_child_weight, subsample = self.subsample, colsample_bytree = self.colsample_bytree,
            n_estimators = self.n_estimators, use_label_encoder =False, scale_pos_weight = self.scale_pos_weight)
            self.hyperParam = ['colsample_bytree', 'eta', 'max_depth', 'min_child_weight', 
            'n_estimators', 'scale_pos_weight', 'subsample']

        elif self.estimator == 'mfnn':
            self.classify = MLPClassifier(
                hidden_layer_sizes = self.hidden_layer_sizes,
                alpha = self.alpha,
                solver = 'adam',
                batch_size = 128,
                max_iter = 100,
                random_state = 42,
                early_stopping = True,
                n_iter_no_change = 5
            )
            self.hyperParam = ['alpha']
            

    def set_params(self, **params):
        for x in params:
            setattr(self, x, params[x])
            
        if 'class_weight' in params:
            params['class_weight'] = {0:params['class_weight'], 1:1-params['class_weight']}

        self.classify.set_params(**params)


    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        return self.classify.fit(X,y)

    def predict(self, X, y=None):
        return self.classify.predict(X)

    def score(self, X, y, sample_weight = None):
        yp = self.predict(X)
        if self.fbeta == None:
            result = f1_score(y, yp, sample_weight = sample_weight, zero_division = 0)
        else:
            result = fbeta_score(y, yp, sample_weight = sample_weight, beta = self.fbeta)
        acc = accuracy_score(y, yp, sample_weight = sample_weight)
        balanced = balanced_accuracy_score(y, yp)
        conf = confusion_matrix(y, yp)

        tempList = list()
        for par in self.hyperParam:
            tempList.append(getattr(self,par))
        filename = os.path.join(self.path, 'results', 'hyperparams_runs', self.estimator + '_' + self.method + '_run_' + str(self.run) + '.txt')
        file = open(filename, 'a')    
        file.write(str(tempList)+':'+str(conf[0][0])+' '+str(conf[0][1])+' '+str(conf[1][0])+' '+str(conf[1][1])+'\n')
        file.close()

        return result

    def makeTuple(self):
        tempList = list()
        for par in self.hyperParam:
            tempList.append(getattr(self,par))
        return tuple(tempList)

class ClassifierHelper():
    '''
    This is the actual class that you will be using, which wraps and automates k-fold cross
    validation with hyperparameter tuning.
    '''
    def __init__(self, train_data, test_data, out_folder):
        '''
        Arguments:
            train_data (DataFrame) -- dataframe to train classifiers on. should only contain feature, target outcome, and ID_1 columns.
            test_data (DataFrame) -- dataframe to test classifiers on. should only contain feature, target outcome, and ID_1 columns.
            out_folder (str) -- path to folder where this instance will write files to. created if doesn't exist.
        '''
        self.train_data = train_data
        self.test_data = test_data
        self.out_folder = out_folder
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        logging.basicConfig(filename= os.path.join(out_folder, 'classifiers.log'), encoding='utf-8', level=logging.DEBUG)


    @staticmethod
    def getParameters(classifier):
        '''
        Get search grid for given classifier. Used for Bayes Search.
        Arguments:
            classifier (str) -- string identifier of classifier. 
        
        Returns
            map (dict(str: range))-- parameter search grid for provided classifier.
            iter (int) -- number of iterations for provided classifier.
        '''
        if classifier =='rdforest':
            return {'n_estimators': Integer(50,2000), 'max_depth': Integer(5,100),
            'min_samples_split':Integer(2,10), 'min_samples_leaf':Integer(1,5),
            'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 60
        elif classifier == 'logreg':
            return {'penalty': ['none'], 'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 30
        elif classifier == 'elasticnet':
            return {'alpha': Real(1e-5, 100, prior = 'log-uniform'),
            'l1_ratio': Real(0.01, 0.99, prior = 'log-uniform'),
            'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 30
        elif classifier == 'naiveBayes':
            return {'alpha': [1]}, 1
        elif classifier == 'knn':
            return {'n_neighbors': Integer(2,10), 'leaf_size': Integer(10,100)}, 30
        elif classifier == 'LDA':
            return {'shrinkage' : Real(0.01, 0.99, prior = 'log-uniform'), 'solver': ['lsqr']}, 30
        elif classifier == 'svm':
            return {'degree': Integer(2, 5), 'gamma':  Real(1e-3, 1e3, prior = 'log-uniform'),
            'C': Real(1e-3, 1e3, prior = 'log-uniform'), 'kernel': ['linear', 'rbf', 'poly'],
            'class_weight': Real(0.01, 0.99, prior = 'log-uniform')}, 60
        elif classifier == 'xgboost':
            return {'eta':Real(0.005, 1, prior = 'log-uniform'), 'max_depth': Integer(3,15),
            'subsample' : Real(0.01, 0.99, prior = 'log-uniform'), 'scale_pos_weight' : Integer(1,100),
            'colsample_bytree' : Real(0.01, 0.99, prior = 'log-uniform'), 
            'n_estimators':Integer(50, 500),  'min_child_weight' : Integer(1,100)}, 80
        elif classifier == 'mfnn':
            return {
                'alpha': Real(1e-6, 1e-1, prior='log-uniform')
            }, 50
        raise Exception('Unknown classifier: {}'.format(classifier))

    def classify(
        self, 
        target_column, 
        n_runs,
        n_folds, 
        classifier, 
        method = 'default',
        balancing_method = None,
        sampling_strategy = None,
        n_test_resamples = 1,
        test_resample_size = 1.0,
        fbeta = None
    ):
        '''
        Wrapper for running k-fold cross validation.

        Arguments:
            target_column (str) -- outcome column in self.data. predictions will be made for this column.
            n_runs (int) -- number of times to run k-fold cross validation.
            n_folds (int) -- number of cross validations in each run (k in k-fold cross validation).
            classifier (str) -- string identifier of the classifier to be used.
            method (str) -- i don't know what this is, but the code might break without it.
            balancing_method ('undersample', 'smote', 'smoten', 'smotenc') -- if provided, will balance training data during k-fold cross validation using provided method. 
            sampling_strategy (float) -- if oversampling or undersampling, will balance data so that target minority class matches this ratio
            n_test_resamples (int) -- test the model after cross validation this many times with random samples from the test split.
            test_resample_size (float) -- randomly samples (without replacements) this fraction from the test data instead of using it entirely during testing.
            fbeta (float) -- if not None, hyperparameter tuning will be performed to optimize F-beta score with this beta value instead of accuracy.
            
        '''

        # sanity checks for parameters
        if (balancing_method is not None) and sampling_strategy is None:
            raise Exception('Missing ratio for data balancing.')

        # prepare feature matrix and outcome vector
        X_train = self.train_data.drop(columns = ['ID_1', target_column]).to_numpy()
        y_train = self.train_data[target_column].to_numpy()

        logging.info('Attempting classification for {} with {}: {} runs of {}-fold cross-validation.'.format(target_column, classifier, n_runs, n_folds))

        # folders to output results in. create them if non-existent
        c_path = os.path.join(self.out_folder, 'results', 'classifiers')
        r_path = os.path.join(self.out_folder, 'results', 'hyperparams_runs')

        if not os.path.exists(c_path):
            os.makedirs(c_path)
        if not os.path.exists(r_path):
            os.makedirs(r_path)

        train_validation_path = os.path.join(c_path, classifier + '_' + method +"_train_validation.txt")
        train_test_path = os.path.join(c_path, classifier + '_' + method + "_train_test.txt")
        open(train_validation_path, 'w').close()
        open(train_test_path, 'w').close()

        for i in range(n_runs):
            result_path = os.path.join(r_path, classifier + "_" + method + '_run_' + str(i + 1) + ".txt")
            open(result_path, "w").close() 
            logging.info('Run {}:'.format(str(i)))

            # create classifier
            myclassif = MyClassifier(
                estimator = classifier, 
                method = method, 
                splits = n_folds, 
                path = self.out_folder, 
                run = i + 1,
                fbeta = fbeta
            )

            # if a balancing method was provided, create an according balancer
            balancer = None
            if balancing_method == 'undersample':
                balancer = RandomUnderSampler(sampling_strategy = sampling_strategy, random_state = 42 + i)     
            elif balancing_method == 'smote':
                balancer = SMOTE(sampling_strategy = sampling_strategy, random_state = 42 + i)
            elif balancing_method == 'smoten':
                # this assumes your data is categorical only!
                balancer = SMOTEN(sampling_strategy = sampling_strategy, random_state = 42 + i)
            elif balancing_method == 'smotenc':
                # grab all categorical columns: assuming they are one-hot-encoded, they should only have 2 or less unique values
                unique_threshold = 2
                X_withnames = self.train_data.drop(columns = ['ID_1', target_column])
                categorical_column_indices = [i for i, column in enumerate(X_withnames.columns) if X_withnames[column].nunique() <= unique_threshold]
                balancer = SMOTENC(
                    categorical_features = categorical_column_indices, 
                    sampling_strategy = sampling_strategy, 
                    random_state = 42 + i
                )
                
            # if a balancer was created, create the cross-validation pipeline with it
            if balancer is None:
                _pipeline = make_pipeline(myclassif)
            else:
                _pipeline = make_pipeline(balancer, myclassif)
            cv = StratifiedKFold(n_splits=n_folds, random_state=i+40, shuffle= True)

            # get hyperparameter search spaces
            parms, itr = ClassifierHelper.getParameters(classifier)
            parms = {'myclassifier__' + key: parms[key] for key in parms}

            # hyperparameter tuning CHANGE THE SCORER 
            grid_imba = BayesSearchCV(
                _pipeline, 
                search_spaces = parms, 
                cv = cv, 
                n_iter = itr, 
                refit = False, 
                n_jobs = -1, 
                n_points = 3,
            )
            grid_imba.fit(X_train, y_train)

            # get best params
            best = grid_imba.best_params_

            # write best params
            bestParam = dict()
            for key in best.keys():
                bestParam[key.split('__')[1]] = best[key]

            if 'class_weight' in bestParam:
                bestParam['class_weight'] = {0:bestParam['class_weight'], 1:1-bestParam['class_weight']}

            confusion = dict()
            with open(result_path, "r") as f:
                for line in f.readlines():
                    params, matrix = line.split(':')[0], line.split(':')[1].split(' ')
                    params = literal_eval(params)
                    matrix = [int(x) for x in matrix]
                    matrix = np.array([[matrix[0], matrix[1]],[matrix[2], matrix[3]]])

                    if tuple(params) in confusion:
                        confusion[tuple(params)].append(matrix)
                    else:
                        confusion[tuple(params)] = [matrix]

            confMatrix = sum(confusion[tuple(best.values())])
            confusion = dict()
            
            trainVal = open(train_validation_path, "a")
            trainVal.write(str(grid_imba.best_params_)+'\n')
            trainVal.write(str(confMatrix).replace(' [', '').replace('[', '').replace(']', '') + '\n\n')
            trainVal.close()

            # create the same classifier with the best parameters from bayes search cv
            if classifier == 'rdforest':
                Tester = RandomForestClassifier(**bestParam)
            elif classifier == 'logreg':
                Tester = LogisticRegression(**bestParam, max_iter=10000)
            elif classifier == 'elasticnet':
                Tester = SGDClassifier(**bestParam, loss = 'log_loss', penalty = 'l1')
            elif classifier == 'naiveBayes':
                Tester = BernoulliNB(**bestParam)
            elif classifier == 'knn':
                Tester = KNeighborsClassifier(**bestParam)
            elif classifier == 'LDA':
                Tester = sk.LinearDiscriminantAnalysis(**bestParam)
            elif classifier == 'svm':
                Tester = SVC(**bestParam, random_state=0, max_iter=10000000)
            elif classifier == 'xgboost':
                Tester = XGBClassifier(**bestParam, use_label_encoder = False)
            elif classifier == 'mfnn':
                Tester = MLPClassifier(**bestParam)

            # train classifier on entire training set
            Tester.fit(X_train, y_train)

            # test classifier for provided times with provided resampling
            for j in range(n_test_resamples):
                size = len(self.test_data) * test_resample_size
                test_data = resample(
                    self.test_data,
                    n_samples = size,
                    random_state = 42 + j,
                    stratify = self.test_data[target_column],
                    replace = False
                )
                X_test = test_data.drop(columns = ['ID_1', target_column]).to_numpy()
                y_test = test_data[target_column].to_numpy()
                
                yp = Tester.predict(X_test)
                fbeta_result = fbeta_score(y_test, yp, beta = fbeta)
                acc = accuracy_score(y, yp)
                balanced = balanced_accuracy_score(y, yp)

                # output results
                trainTest = open(train_test_path, "a")
                confMatrix = confusion_matrix(y_test,yp)
                trainTest.write('Run {}, test sample {}:'.format(str(i), str(j)))
                trainTest.write('F-beta score: {:.4f}\n'.format(fbeta_result))
                trainTest.write('Accuracy: {:.4f}\n'.format(acc))
                trainTest.write('Balanced accuracy: {:.4f}\n'.format(balanced))
                trainTest.write(str(grid_imba.best_params_)+'\n')
                trainTest.write(str(confMatrix).replace(' [', '').replace('[', '').replace(']', '') + '\n\n')
                trainTest.close()