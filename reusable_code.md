# reusable code from past projects
## Data Preparation  
It is best to refer to original `ukbb_parser` documentation at https://github.com/nadavbra/ukbb_parser/blob/master/ukbb_parser/ukbb_parser.py
* **Non-genetic data (e.g. clinical factors):** available on `load_data.py` (Hieu)
    * import: `ukbb_parser.create_dataset`
* **Non-imputed genetic data:** available on `load_data.py` (Hieu)
    * imports: `ukbb_parser.get_chrom_raw_marker_data`, `pandas_plin
## Feature Selection
* **$χ^2$ test:** available on `RunFeatureSelection.py` (Cole), line 39  
    * import: `sklearn.feature_selection.chi2` 
```
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
```
* **InfoGain test:** available on `RunFeatureSelection.py` (Cole), line 187  
    * import: `sklearn.feature_selection.mutual_info_classif`  
```
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
```
* **Maximum Relevance Minimum Redundancy (MRMR) test:** available on `RunFeatureSelection.py` (Cole), line 56  
    * import: `LCSI.lcsi`
```
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
```  
* **Joint Mutual Information (JMI) test:** available on `RunFeatureSelection.py` (Cole), line 89  
    * import: `LCSI.lcsi`
```
def jmi(X, y, **kwargs):
    """
    This function implements the JMI feature selection
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
        F, J_CMI, MIfy = LCSI.lcsi(X, y, function_name='JMI', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = LCSI.lcsi(X, y, function_name='JMI')
    return F
```   
## Statistical Analysis
* **Mediation Analysis:** available on `StatisticalAnalysis.py` (Cole), line 188  
    * import: `rpy2`
    * additional required files: `mediationAnalysis.R`
```
def Mediation_Analysis(self, dep, mediator, indep, sims=1000):
        '''
        Do Mediation Analysis between the passed dependent & indepent variables 
        and the mediation variable(s) passed in the passed data frame. 
        Write the results to Mediation_Analysis.txt in the passed path
        
        Args:
            data: DataFrame containing the items of interest
            dep: The dependent varaible in the analysis
            mediator: The mediation variable in the analysis
            indep: The independent variable(s) in the analysis - can be a list
            continuous: list containing continuous variables
        '''
        data = self.dpOHCnonBin.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpOHCnonBin.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])


        if not os.path.exists(self.statsPath + 'MediationAnalysis/'):
            os.makedirs(self.statsPath +'MediationAnalysis/')
        currPath = self.statsPath + 'MediationAnalysis/'

        if type(indep) == str:
            t = list(); t.append(indep)
            indep = t

        for var in indep:
            filePath = currPath+'MedAnalysis-'+str(var) + '-' + str(mediator) + '-' + str(dep) + '.txt'

            l1 = importr('mediation')
            formulaMake = r['as.formula']
            mediate, lm, glm, summary, capture = r['mediate'], r['lm'], r['glm'], r['summary'], r['capture.output']

            MediationFormula = formulaMake(mediator + ' ~ ' + var)
            OutcomeFormula = formulaMake(dep + ' ~ ' + var + ' + ' + mediator)

            with localconverter(ro.default_converter + pandas2ri.converter):
                data = ro.conversion.py2rpy(data)

            if mediator in continuous:
                modelM = lm(MediationFormula, data) 
            else:
                modelM = glm(MediationFormula, data = data, family = "binomial")
            
            if dep in continuous:
                modelY = lm(OutcomeFormula, data)
            else:
                modelY = glm(OutcomeFormula, data = data, family = "binomial")
            
            results = mediate(modelM, modelY, treat=var, mediator=mediator,sims=sims)
            dfR = summary(results)
            self.mediation['results'] = dfR
            capture(dfR, file = filePath)
```  
* **Association Rule Learning:** available on `StatisticalAnalysis.py` (Cole), line 52
    * import: `rpy2`
    * additional required files: `Association_Rules.R`
```
def Association_Rule_Learning(self, rhs, min_support = 0.00045, min_confidence = 0.02, min_items = 2, max_items = 5, min_lift = 2, protective = False):
        ''' 
        Do Association Rules mining for the items within the passed dataframe. Write all the found 
        association rules that meet the specified conditions and save the produced graphs
        in the passed parameters to AssociationRules.txt in Apriori Folder in the passed path.
        
        Args:
            data: DataFrame containing the items of interest
            min_support: Minimum value of support for the rule to be considered
            min_confidence: Minimum value of confidence for the rule to be considered
            min_items: Minimum number of item in the rules including both sides
            max_items: Maximum number of item in the rules including both sides
            rhs: Item to be on the right hand side -- outcome variable
            min_lift: Minimum value for lift for the rule to be considered
            protective: If True, the rhs values will be flipped to find protective features
        Returns:
            A dataframe of all the association rules found
        '''

        data = self.dpOHC.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        columns=data.columns
        for feature in columns:
            if '.' in feature:
                data.drop([feature],axis=1,inplace=True)
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpOHC.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])
        data.drop(labels=continuous,axis=1,inplace=True)

        for col in data.columns:
            if len(data[col].unique()) > 2:
                data.drop([col], axis = 1, inplace = True)
        
        print(data)
        if not os.path.exists(self.statsPath+'Apriori'):
            os.makedirs(self.statsPath+'Apriori')
        aprioriPath = self.statsPath + 'Apriori/'
        
        
        if protective:
            data = data.copy(deep = True)
            data[rhs] = np.absolute(np.array(data[rhs].values)-1)
            
            if not os.path.exists(aprioriPath+'Protective/'):
                os.makedirs(aprioriPath+'Protective/')
            currPath = aprioriPath+'Protective/'
        
        else:
            if not os.path.exists(aprioriPath+'Risk/'):
                os.makedirs(aprioriPath+'Risk/')
            currPath = aprioriPath+'Risk/'

        data.to_csv(currPath + 'AprioriData.csv')
        args = currPath + ' ' + str(min_support) + ' '  + str(min_confidence) + ' ' + str(max_items) + ' ' + str(min_items) + ' ' + str(rhs) + ' ' + str(min_lift)
        os.system('Rscript ' + self.path + 'Association_Rules.R ' + args)
        os.remove(currPath + 'AprioriData.csv') 
        if os.path.exists(currPath + 'apriori.csv'):
            ARLRules = pd.read_csv(currPath + 'apriori.csv')
            pvals = ARLRules['pValue']
            pvals = fdr_correction(pvals, alpha=0.05, method='indep')
            ARLRules['adj pVals'] = pvals[1]
        else:
            print('No rules meeting minimum requirements were found')
            print('Process Terminated')
            return
        os.remove(currPath + 'apriori.csv') 

        vars = ARLRules['LHS'].tolist()
        features, newF, rows, pvals = list(), list(), list(), list()
        oddsRatios = pd.DataFrame(columns=['LHS-RHS', 'Odds Ratio', 'Confidence Interval', 'pValue', 'adjusted pVal'])
        for var in vars:
            newF.append(var)
            features.append(var.replace('{', '').replace('}', '').split(','))
        for i in range(len(features)):
            cols = features[i]
            newFeature = newF[i]
            dataC = data.drop([x for x in data.columns if x not in cols], axis = 1)
            dataC[newFeature] = dataC[dataC.columns[:]].apply(lambda x: ','.join(x.astype(str)),axis=1)
            dataC = dataC[[newFeature]]
            dataC[rhs] = data[rhs]
            toDrop = list()
            for index, r in dataC.iterrows():                
                fValue = set(r[newFeature].split(','))
                if (len(fValue) > 1):
                    toDrop.append(index)
            dataC.drop(toDrop, inplace = True)
            dataTrue = dataC[dataC[rhs] == 1].drop([rhs], axis =1).value_counts().tolist()
            dataFalse = dataC[dataC[rhs] == 0].drop([rhs], axis = 1).value_counts().tolist()
            if len(dataTrue) == 1:
                dataTrue.append(0)
            if len(dataFalse) == 1:
                dataFalse.append(0)
            dataTrue.reverse(); dataFalse.reverse()
            
            table = np.array([dataTrue, dataFalse])
            print(table)
            res = statsmodels.stats.contingency_tables.Table2x2(table, shift_zeros = True)

            rows.append([str(newFeature)+'-'+str(rhs), 
            res.oddsratio, res.oddsratio_confint(), res.oddsratio_pvalue()])
            pvals.append(res.oddsratio_pvalue())
            
        pvals = fdr_correction(pvals, alpha=0.05, method='indep')
        for i in range(len(pvals[1])):
            rows[i].append(pvals[1][i])
            oddsRatios.loc[len(oddsRatios.index)] = rows[i]
            
            

        Association = open(currPath + 'AssociationRules.txt', 'w')
        Association.write(ARLRules.to_string(index=False))
        Association.write('\n\n----------------------------------------------------------------------------------------\n\n')
        Association.write('\n\nOdds Ratio analysis for Association Rule Learning: \n----------------------------------------------------------------------------------------\n\n')
        for i in range(len(oddsRatios)):
            two_var = oddsRatios.iloc[i, :]
            two_var = two_var.to_frame()
            variables = str(two_var.iloc[0,0]).split('-')
            two_var = two_var.iloc[1: , :]
            Association.write('The odds ratio, p-Value, and confidence interval between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
            toWrite = two_var.to_string(header = False, index = True)
            Association.write(toWrite+'\n')
            Association.write('----------------------------------------------------------------------------------------\n\n')

        Association.close()
        if protective:
            self.ARL['ARLProtect'] = ARLRules
        else:
            self.ARL['ARLRisk'] = ARLRules

        
        
        return ARLRules
```  
* **Two-way ANOVA:** available on `StatisticalAnalysis.py` (Cole), line 1226
    * import: `bionfokit.analys`  
    (there is way too much going on here, so check the actual code)
```
def ANOVA(self, dep, indep, alpha = 0.05, oneWay = True, followUp = False, oneVsOther = dict(), oneVsAnother =dict()):
        '''
        Conduct an ANOVA analysis -- either one or two way -- between the dependent and independent variables
        passed. If there is signifcant effect found, conduct a follow up test. The function checks for the ANOVA
        assumption and provide alternative tests such as Kruskal-Wallis H. Results will be stored at ###
        
        Args:
            data: DataFrame containing the items of interest
            dep: the dependent varaible in the analysis
            indep: column names in the data frame containing the groups -- can be 
                a string or a list of two strings
            alpha: minimum value for the p-value for the effect to be signifcant
                conduct repeated measures ANOVA -- to be implemented.
            oneWay: if True, conduct one way ANOVA. if False, conduct two way ANOVA.
            followUp: if True, a follow up test would be conducted regardless of the ANOVA p-value

        Returns:
            a dictionary mapping each test conducted to its results
        '''
        data = self.dpnonOHCnonBin.df.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        types=np.array(self.dpnonOHCnonBin.types)
        features=np.array(data.columns)
        continuous=list(features[np.where(types!='c')[0]])
        if dep in continuous:
            continuous.remove(dep)
        data.drop(labels=continuous,axis=1,inplace=True)
        OGdata = data.copy(deep = True)

        for var in oneVsOther.keys():
            data.loc[data[var] != oneVsOther[var], var] = 'other'
            #newName = var+'_oneVsRest_'+str(oneVsOther[var])
            newName = var+'_'+str(oneVsOther[var])
            data.rename(columns = {var:newName}, inplace = True)
            if indep == var:
                indep = newName
            elif var in indep:
                indep.remove(var)
                indep.append(newName)

        for var in oneVsAnother.keys():

            d1 = data[data[var] == oneVsAnother[var][0]]
            d2 = data[data[var] == oneVsAnother[var][1]]
            data = pd.concat([d1,d2])
            newName = var+'_'+str(oneVsAnother[var][0])+'Vs'+str(oneVsAnother[var][1])
            data.rename(columns = {var:newName}, inplace = True)
            if indep == var:
                indep = newName
            elif var in indep:
                indep.remove(var)
                indep.append(newName)

        if oneWay:
            if (type(indep) == list):
                indep = indep[0]
            if not oneVsOther:
                data = data.astype({indep: 'int64'})
            else:
                data = data.astype({indep: 'str'})
            try:    
                return self.oneWay_ANOVA(data, dep, indep, alpha, False, followUp)
            except ValueError:
                return
        else:
            if not oneVsOther:
                data = data.astype({indep[0]:'int64'})
                data = data.astype({indep[1]:'int64'})
            else:
                data = data.astype({indep[0]:'str'})
                data = data.astype({indep[1]:'str'})
            return self.twoWay_ANOVA(data, dep, indep, alpha, False, followUp, OGdata)
```
## Classification
* All classifier code can be found in `RunClassifiers.py` (Cole)
    * imports: `sklearn`, `imblearn`, `xgboost`  
(again, a lot going on here so we need to go over the code)

## Regression
* **Multivariate Logistic Regression:** available on `main.py` (Hieu), line 30
    * import: `statsmodels` (as `sm`)
```
def mvlog(Y_train, x_train, result_file):
    X_train_sm = sm.add_constant(x_train)
    logm2 = sm.GLM(Y_train, X_train_sm, family=sm.families.Binomial())
    res = logm2.fit()
    display_result(X_train_sm.columns, res, result_file)
```  
* **Multivariate Linear Regression:** available on `main.py` (Hieu), line 38  
    * import: `statsmodels` (as `sm`)
```
def mvlinear(Y_train, x_train, result_file):
    X_train_sm = sm.add_constant(x_train)
    logm2 = sm.GLM(Y_train, X_train_sm, family=sm.families.Gaussian())
    res = logm2.fit()
    display_result(X_train_sm.columns, res, result_file)
```
