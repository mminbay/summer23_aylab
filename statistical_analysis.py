import os
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import pandas as pd
import statsmodels
import statsmodels.api as sm
import bioinfokit.analys
from mne.stats import fdr_correction
import researchpy as rp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.genmod import families
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri
numpy2ri.activate()
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import itertools
from utilities import relabel  # hardcoded. relabel functional will be updated when anova needed
from scipy.stats import sem
from scipy import stats
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx

'''
This class is meant to take care of all your statistical analysis. 
An example usage can be found at the end of this file
'''

class Stat_Analyzer():
    # TODO: implement!
    def __init__(
        self,
        binOHCdata,
        binOHCdata_withBase,
        nonbinOHCdata,
        binnonOHCdata,
        nonbinnonOHCdata,
        r_path,
        out_folder
    ):
        '''
        Arguments:
            binOHCdata -- dataframe to be analyzed, where categorical features are OHC'd and outcome is binarized
            binOHCdata_withBase -- dataframe that's the same as binOHCdata, but does not have the baseline for the OHC removed. Used in ARL only
            nonbinOHCdata -- dataframe to be analyzed, where categorical features are OHC'd and outcome is continuous
            binnonOHCdata -- dataframe to be analyzed, where categorical features are not OHC'd and outcome is binarized
            nonbinnonOHCdata -- dataframe to be analyzed, where categorical features are not OHC'd and outcome is continuous
            r_path -- path to directory containing Rscripts that are needed for some statistical analysis
            out_folder -- where this instance will output results

        '''
        self.binOHCdata = binOHCdata
        self.binOHCdata_withBase = binOHCdata_withBase
        self.nonbinOHCdata = nonbinOHCdata
        self.binnonOHCdata = binnonOHCdata
        self.nonbinnonOHCdata = nonbinnonOHCdata
        self.r_path = r_path
        self.out_folder = out_folder

        self.ARL = dict()
        self.mediation = dict()
        self.linearRegression = dict()
        self.logisticRegression = dict()
        self.oneANOVA = dict()
        self.twoANOVA = dict()
    
    def anova(
        self,
        dep,
        indep,
        continuous = [],
        alpha = 0.05,
        one_way = True,
        follow_up = False,
        one_vs_other = dict(),
        one_vs_another = dict()
    ):
        '''
        Conduct an ANOVA analysis -- either one or two way -- on continuous, non OHC'd data between the dependent
        and independent variables passed. If there is signifcant effect found, conduct a follow up test. The function
        checks for the ANOVA assumption and provide alternative tests such as Kruskal-Wallis H.

        Arguments:
            dep -- dependent variable, or outcome label of your study
            indep -- independent variable, or list of two independent variables (depending on one-way or two-way)
            continuous -- list of continuous variables. these will be dropped
            alpha -- minimum value for the p-value for the effect to be significant
            one_way -- if True, conduct one-way ANOVA. otherwise, conduct two-way ANOVA
            follow_up -- if True, conduct follow up tests regardless of p-value
            one_vs_other -- i dont know
            one_vs_another -- i dont know either
        '''
        data = self.nonbinnonOHCdata.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')
        data.drop(columns = continuous, inplace = True)
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
                return self.__one_way_anova(data, dep, indep, alpha, False, follow_up)
            except ValueError:
                return
        else:
            if not oneVsOther:
                data = data.astype({indep[0]:'int64'})
                data = data.astype({indep[1]:'int64'})
            else:
                data = data.astype({indep[0]:'str'})
                data = data.astype({indep[1]:'str'})
            return self.__two_way_anova(data, dep, indep, alpha, False, follow_up, OGdata)

    def __one_way_anova(self, data, dep, indep, alpha, between, followUp):
        results = dict()
        one_way_anova_folder = os.path.join(self.out_folder, 'one_way_ANOVA')
        if not os.path.exists(one_way_anova_folder):
            os.makedirs(one_way_anova_folder)

        print(indep)
        #new_indep=self.relabel(indep[:indep.find('_oneVsRest')+1]+indep[indep.find('oneVsRest')+len('oneVsRest')+1:])
        new_indep=relabel(indep)
        print(new_indep)
        #new_indep=new_indep[:new_indep.rfind('_')+1]+'oneVsRest'+new_indep[new_indep.rfind('_'):]
        data2=data.rename(columns={indep:new_indep},inplace=False)
        indep=new_indep

        indep_folder = os.path.join(one_way_anova_folder, indep)
        if not os.path.exists(indep_folder):
            os.makedirs(indep_folder)

        formula = dep + ' ~ C(' + indep + ')'
        oneWayANOVA = open(os.path.join(indep_folder, 'oneWayANOVA_summary.txt'), 'w')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        
        #Create a box plot for outliers detection
        
        data = data2[[indep, dep]]
        colors = ['#808080']
        box = sns.boxplot(x=indep, y=dep, data=data, palette=colors)
        fig = box.get_figure()
        fig.savefig(os.path.join(indep_folder, 'oneWayANOVA_boxPlot.png'))
        plt.close()
        
        #Create a bar plot

        sns.set(rc = {'figure.figsize':(15,10)})
        sns.set(font_scale = 1.5)
        sns.set_style('whitegrid')
        fig, bar = plt.subplots()
        
        colors = ['#808080']
        
        if data2[indep[0]].isin(['other']).any():
            for i in range(len(data2)):
                if data2[indep[0]].iat[i]!='other':
                    data2[indep[0]].iat[i]=indep[0][indep[0].rfind('_')+1:]

        sns.barplot(x=indep, ax = bar, y=dep, data=data2, palette = colors, capsize=.1)
        width = 0.3

        num_var2 = len(data[indep].unique())
        hatches = itertools.cycle(['+', 'x', '-', '/', '//', '\\', '*', 'o', 'O', '.'])

        '''
        for i, patch in enumerate(bar.patches):
            # Set a different hatch for each bar
            if i % num_var2 == 0:
                hatch = next(hatches)
            patch.set_hatch(hatch)
        '''

        for patch in bar.patches:
            current_width = patch.get_width()
            diff = current_width - width
            patch.set_width(width)
            patch.set_x(patch.get_x() + diff * .5)
            patch.set_edgecolor('#000000')
        fig = bar.get_figure()
        fig.savefig(os.path.join(indep_folder, 'oneWayANOVA_barPlot.png'))
        plt.close()

        #Conducting the ANOVA test
        oneWayANOVA.write('Results for one way ANOVA between ' + indep + ' and ' + dep + ' are: \n\n')
        res = bioinfokit.analys.stat()
        res.anova_stat(df=data, res_var=dep, anova_model=formula)
        asummary = res.anova_summary.to_string(header=True, index=True)
        oneWayANOVA.write(asummary + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['ANOVA_Results'] = res.anova_summary

        #Follow-up Test TukeyTest
        if (res.anova_summary.iloc[0,4] > alpha) and (not followUp):
            oneWayANOVA.write('The p-value is higher than alpha; hence, no follow-up test was conducted\n')
        else:
            oneWayANOVA.write('Results for follow-up Tukey test between ' + indep + ' and ' + dep + ' are: \n\n')
            if len(data[indep].value_counts()) <= 2:
                oneWayANOVA.write('Only two groups. No follow up test was conducted\n')
            else:
                followUp = bioinfokit.analys.stat()
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep, anova_model=formula)
                fSummary = followUp.tukey_summary.to_string(header=True, index=True)
                oneWayANOVA.write(fSummary + '\n')
                results['Tukey_Results'] = followUp.tukey_summary
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
            

        #histograms and QQ-plot for Normality detection
        sm.qqplot(res.anova_std_residuals, line='45')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Standardized Residuals")
        plt.savefig(os.path.join(indep_folder, 'oneWayANOVA_qqPlot.png'))
        plt.close()

        plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
        plt.xlabel("Residuals")
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(indep_folder, 'oneWayANOVA_histogram.png'))
        plt.close()

        #Shapiro-Wilk Test for Normality
        w, pvalue = stats.shapiro(res.anova_model_out.resid)
        oneWayANOVA.write('Results for Shapiro-Wilk test to check for normality are: \n\n')
        oneWayANOVA.write('w is: ' + str(w) + '/ p-value is: ' + str(pvalue) + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Shaprio-Wilk_Results'] = (w, pvalue)

        #Check for equality of varianve using Levene's test
        oneWayANOVA.write("Results for Levene's test to check for equality of variance are: \n\n")
        eqOfVar = bioinfokit.analys.stat()
        eqOfVar.levene(df=data, res_var=dep, xfac_var=indep)
        eqOfVarSummary = eqOfVar.levene_summary.to_string(header=True, index=False)
        oneWayANOVA.write(eqOfVarSummary + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Levene_Results'] = eqOfVar.levene_summary

        #The Kruskal-Wallis H test
        groups = list()
        vals = data[indep].unique()
        for val in vals:
            g = data.loc[data[indep] == val]
            g = g.loc[:, [dep]].squeeze().tolist()
            groups.append(g)

        Kruskal = stats.kruskal(*groups)
        oneWayANOVA.write('Results for the Kruskal-Wallis Test -- to be used if ANOVA assumptions are violated: \n\n')
        oneWayANOVA.write('statistic: ' + str(Kruskal[0]) + '/ p-value is: ' + str(Kruskal[1]) + '\n')
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Kruskal-Wallis_Results'] = Kruskal

        
        #The dunn's test -- follow up
        if (Kruskal[1] > alpha) and (not followUp):
            oneWayANOVA.write('The p-value is higher than alpha; hence, no follow-up test was conducted for Kruskal test\n')    
        else:
            oneWayANOVA.write("Results for follow-up Dunn's test between " + indep + " and " + dep + " are: \n\n")
            if len(data[indep].value_counts()) <= 2:
                oneWayANOVA.write('Only two groups. No follow up test was conducted\n')
            else:
                FSA = importr('FSA')
                dunnTest, formulaMaker, names = r['dunnTest'], r['as.formula'], r['names']
                with localconverter(ro.default_converter + pandas2ri.converter):
                    rDf = ro.conversion.py2rpy(data)

                formula = formulaMaker(dep + ' ~ ' + indep)
                dunnTwoWay = dunnTest(formula, data=rDf, method="bonferroni")

                asData, doCall, rbind = r['as.data.frame'], r['do.call'], r['rbind']
                dunnTwoWay = asData(doCall(rbind, dunnTwoWay))

                with localconverter(ro.default_converter + pandas2ri.converter):
                    dunnTwoWay = ro.conversion.rpy2py(dunnTwoWay)

                dunnTwoWay.drop(['method', 'dtres'], inplace = True)

                for col in ['Z', 'P.unadj', 'P.adj']:
                    dunnTwoWay[col] = pd.to_numeric(dunnTwoWay[col])
                    dunnTwoWay[col] = np.round(dunnTwoWay[col], decimals = 5)
                
                dunnSummary = dunnTwoWay.to_string(header=True, index=False)
                oneWayANOVA.write(dunnSummary + '\n\n')
        
        oneWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        oneWayANOVA.close()

        self.oneANOVA = results
        return results
    
    def __two_way_ANOVA(self, data, dep, indep, alpha, between, followUp, OGdata):
        results = dict()
        if len(data[indep[0]].value_counts()) < len(data[indep[1]].value_counts()):
            temp = indep.pop(0)
            indep.append(temp)
        elif len(data[indep[0]].value_counts()) == len(data[indep[1]].value_counts()):
            temp1 = indep.pop(0)
            temp2 = indep.pop(0)
            if len(temp1) <= len(temp2):
                indep.append(temp2); indep.append(temp1)
            else:
                indep.append(temp1); indep.append(temp2)
        
        print(indep)
        new_indep=relabel(indep[0])
        data=data.rename(columns={indep[0]:new_indep},inplace=False)
        OGdata=OGdata.rename(columns={indep[0][:indep[0].find('_')]:relabel(indep[0][:indep[0].find('_')]),indep[0][indep[0].find('_')+1:indep[0].rfind('_')]:relabel(indep[0][indep[0].find('_')+1:indep[0].rfind('_')])},inplace=False)
        indep=[new_indep,indep[1]]
        print(indep)

        two_way_anova_folder = os.path.join(self.out_folder, 'two_way_ANOVA')
        if not os.path.exists(two_way_anova_folder):
            os.makedirs(two_way_anova_folder)
        result_name = indep[0] + '_' + indep[1]
        result_folder = os.path.join(two_way_anova_folder, result_name)
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        formula = dep + ' ~ C(' + indep[0] + ') + C(' + indep[1] + ') + C(' + indep[0] + '):C(' + indep[1] + ')'
        twoWayANOVA = open(os.path.join(result_folder, 'twoWayANOVA_summary.txt'), 'w')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
            
        #Create a box plot
        data = data[indep + [dep]]
        colors = ['#808080', '#FFFFFF', '#C0C0C0']
        box = sns.boxplot(x=indep[0], y=dep, hue=indep[1], data=data, palette = colors, width = 0.6)
        fig = box.get_figure()
        fig.savefig(os.path.join(result_folder, "twoWayANOVA_boxPlot.png"))
        plt.close()

        #Create a bar plot
        sns.set(rc = {'figure.figsize':(15,10)})
        sns.set(font_scale = 1.5)
        sns.set_style('whitegrid')

        if data[indep[0]].isin(['other']).any():
            for i in range(len(data)):
                if data[indep[0]].iat[i]!='other':
                    data[indep[0]].iat[i]=indep[0][indep[0].rfind('_')+1:]

        if 'sex' in indep:
            data['sex']=data['sex'].astype(str)
            for i in range(len(data)):
                if data[indep[1]].iat[i]=='0.0':
                    data[indep[1]].iat[i]='female'
                else:
                    data[indep[1]].iat[i]='male'
        
        # figure out what this is
        # self.plotComboBox(indep[0],dep,data,OGdata,currPath)
        fig, bar = plt.subplots()
        
        colors = ['#808080', '#FFFFFF', '#C0C0C0']

        sns.barplot(x=indep[0], ax = bar, y=dep, hue=indep[1], data=data, 
        palette = colors, capsize=.1)
        width = 0.3

        num_var2 = len(data[indep[0]].unique())
        hatches = itertools.cycle(['+', 'x', '-', '/', '//', '\\', '*', 'o', 'O', '.'])

        '''
        for i, patch in enumerate(bar.patches):
            # Set a different hatch for each bar
            if i % num_var2 == 0:
                hatch = next(hatches)
            patch.set_hatch(hatch)
        '''

        for patch in bar.patches:
            current_width = patch.get_width()
            diff = current_width - width
            patch.set_width(width)
            patch.set_x(patch.get_x() + diff * .5)
            patch.set_edgecolor('#000000')

        bar.legend(frameon = 1, title = indep[1], fontsize = 15, title_fontsize = 20,
        bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)     
        fig = bar.get_figure()
        fig.savefig(os.path.join(result_folder, "twoWayANOVA_barPlot.png"))
        plt.close()
            
        #Conducting the ANOVA test
        twoWayANOVA.write('Results for two way ANOVA between ' + indep[0] + '&' + indep[1] + ' and ' + dep + ' are: \n\n')
        res = bioinfokit.analys.stat()
        res.anova_stat(df=data, res_var=dep, anova_model=formula, ss_typ=3)
        asummary = res.anova_summary.iloc[1:, :].to_string(header=True, index=True)
        twoWayANOVA.write(asummary + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['ANOVA_Results'] = res.anova_summary

        #Follow-up Test TukeyTest
        if (all(x > alpha  for x in res.anova_summary.iloc[1:4, 4].tolist())) and (not followUp):
            twoWayANOVA.write('All the p-values is higher than alpha; hence, no follow-up test was conducted\n\n')
        else:
            tukey = list()
            message = list()
            message.append('Main effect for ' + indep[0] + ':\n')
            message.append('Main effect for ' + indep[1] + ':\n')
            message.append('Interaction effect between ' + indep[0] + ' and ' + indep[1] + ':\n')                
            twoWayANOVA.write('Results for follow-up Tukey test between ' + indep[0] + ' & ' + indep[1] + ' and ' + dep + ' are: \n\n')
            followUp = bioinfokit.analys.stat()
            if len(data[indep[0]].value_counts()) <= 2:
                tukey.append('Only two groups. No follow up test was conducted\n')
            else:
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep[0], anova_model=formula)
                tukey.append(followUp.tukey_summary.to_string(header=True, index=False))
            
            if len(data[indep[1]].value_counts()) <= 2:
                tukey.append('Only two groups. No follow up test was conducted\n')
            else:
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep[1], anova_model=formula)
                tukey.append(followUp.tukey_summary.to_string(header=True, index=False))

            if len(data[indep[0]].value_counts())*len(data[indep[1]].value_counts()) <= 2:
                tukey.append('Only two groups. No follow up test was conducted\n')
            elif res.anova_summary.iloc[1:4, 4].tolist()[2] > alpha:
                tukey.append('Interaction effect not significant. No follow up test was conducted\n')
            else:
                followUp.tukey_hsd(df=data, res_var=dep, xfac_var=indep, anova_model=formula)
                tukey.append(followUp.tukey_summary.to_string(header=True, index=False))

            for i in range(len(tukey)):
                twoWayANOVA.write(message[i] + tukey[i] + '\n\n')
                twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
            results['Tukey_Results'] = tukey
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
                
        #histograms and QQ-plot
        sm.qqplot(res.anova_std_residuals, line='45')
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Standardized Residuals")
        plt.savefig(os.path.join(result_folder, 'twoWayANOVA_qqPlot.png'))
        plt.close()

        plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
        plt.xlabel("Residuals")
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(result_folder, 'twoWayANOVA_histogram.png'))
        plt.close()
                
        #Shapiro-Wilk Test for Normality
        w, pvalue = stats.shapiro(res.anova_model_out.resid)
        twoWayANOVA.write('Results for Shapiro-Wilk test to check for normality are: \n\n')
        twoWayANOVA.write('w is: ' + str(w) + '/ p-value is: ' + str(pvalue) + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Shaprio-Wilk_Results'] = (w, pvalue)
            
        #Check for equality of varianve using Levene's test
        twoWayANOVA.write("Results for Levene's test to check for equality of variance are: \n\n")
        eqOfVar = bioinfokit.analys.stat()
        eqOfVar.levene(df=data, res_var=dep, xfac_var=indep)
        eqOfVarSummary = eqOfVar.levene_summary.to_string(header=True, index=False)
        twoWayANOVA.write(eqOfVarSummary + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['Levene_Results'] = eqOfVar.levene_summary

        #The scheirer Ray Hare test

        rcompanion = importr('rcompanion')
        formulaMaker = r['as.formula']
        scheirerRayHare = r['scheirerRayHare']

        formula = formulaMaker(dep + ' ~ '  + indep[0] + ' + ' + indep[1])
        print(data.dtypes)
        with localconverter(ro.default_converter + pandas2ri.converter):
                rDf = ro.conversion.py2rpy(data)

        scheirerANOVA = scheirerRayHare(formula, data = rDf)

        with localconverter(ro.default_converter + pandas2ri.converter):
            scheirerANOVA = ro.conversion.rpy2py(scheirerANOVA)

        twoWayANOVA.write('Results for the scheirer Ray Hare Test -- to be used if ANOVA assumptions are violated: \n\n')
        scheirerSummary = scheirerANOVA.to_string(header=True, index=True)
        twoWayANOVA.write(scheirerSummary + '\n')
        twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')
        results['scheirerRayHare'] = scheirerANOVA
        
        #The dunn's test -- follow up
        if (all(x > alpha  for x in scheirerANOVA['p.value'].tolist())) and (not followUp):
            twoWayANOVA.write('All the p-values is higher than alpha; hence, no follow-up test was conducted for ScheirerRayHare test\n\n')
        
        else:
            FSA = importr('FSA')
            dunnTest = r['dunnTest']
            data['interaction'] = data[indep[0]].astype(str) + '_' + data[indep[1]].astype(str)
            with localconverter(ro.default_converter + pandas2ri.converter):
                rDf = ro.conversion.py2rpy(data)
            indep.append('interaction')            
            for var in indep:
                formula = formulaMaker(dep + ' ~ ' + var)
                dunnTwoWay = dunnTest(formula, data=rDf, method="bonferroni")

                asData, doCall, rbind = r['as.data.frame'], r['do.call'], r['rbind']
                dunnTwoWay = asData(doCall(rbind, dunnTwoWay))

                with localconverter(ro.default_converter + pandas2ri.converter):
                    dunnTwoWay = ro.conversion.rpy2py(dunnTwoWay)

                dunnTwoWay.drop(['method', 'dtres'], inplace = True)
                for col in ['Z', 'P.unadj', 'P.adj']:
                    dunnTwoWay[col] = pd.to_numeric(dunnTwoWay[col])
                    dunnTwoWay[col] = np.round(dunnTwoWay[col], decimals = 5)
                if var == 'interaction':
                    twoWayANOVA.write("Results for follow-up Dunn's test between " + indep[0] + ' & ' + indep[1] + " and " + dep + " are: \n\n")
                else:
                    twoWayANOVA.write("Results for follow-up Dunn's test between " + var + " and " + dep + " are: \n\n")
                
                if len(data[var].value_counts()) <= 2:
                    twoWayANOVA.write('Only two groups. No follow up test was conducted\n')
                elif var == 'interaction' and scheirerANOVA['p.value'].tolist()[2] > alpha:
                    twoWayANOVA.write('Interaction effect not significant. No follow up test was conducted\n')
                else:
                    dunnSummary = dunnTwoWay.to_string(header=True, index=False)
                    twoWayANOVA.write(dunnSummary + '\n\n')
        
                twoWayANOVA.write('----------------------------------------------------------------------------------------\n\n')

        self.twoANOVA = results
        return results

    def mediation_analysis(
        self,
        dep,
        med,
        indep,
        continuous,
        sims = 1000
    ):
        '''
        Conducts mediation analysis on continous, OHC data for given dependent variable, mediator, and independent variable(s).
        Saves results as a .csv file

        Arguments:
            dep -- dependent variable, or outcome label of your study
            med -- mediator variable
            indep -- independent variable, or list of independent variables
            continuous -- list of continuous 
            sims -- i don't even know
        '''
        data = self.nonbinOHCdata.copy(deep = True)

        result_folder = os.path.join(self.out_folder, 'mediation_analysis')

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        # convert indep to a list if a single var was passed
        if type(indep) == str:
            t = list(); t.append(indep)
            indep = t
        
        for var in indep:
            current_var_file = str(var) + '-' + str(med) + '-' + str(dep) + '.txt'
            result_file = os.path.join(result_folder, current_var_file)

            # l1 = importr('mediation')
            # formulaMake = r['as.formula']
            # mediate, lm, glm, summary, capture = r['mediate'], r['lm'], r['glm'], r['summary'], r['capture.output']

            # MediationFormula = formulaMake(mediator + ' ~ ' + var)
            # OutcomeFormula = formulaMake(dep + ' ~ ' + var + ' + ' + mediator)
            mediation_formula = mediator + ' ~ ' + var
            outcome_formula = dep + ' ~ ' + var + ' + ' + mediator

            # with localconverter(ro.default_converter + pandas2ri.converter):
            #     data = ro.conversion.py2rpy(data)

            if med in continuous:
                # modelM = lm(MediationFormula, data)
                model_m = sm.OLS.from_formula(mediation_formula, data)
                
            else:
                modelM = sm.GLM(MediationFormula, data = data, family = "binomial")
            
            if dep in continuous:
                modelY = lm(OutcomeFormula, data)
            else:
                modelY = glm(OutcomeFormula, data = data, family = "binomial")
            
            results = mediate(modelM, modelY, treat=var, mediator=mediator,sims=sims)
            dfR = summary(results)
            self.mediation['results'] = dfR
            capture(dfR, file = result_file)

    # ARL below was written mostly by chatgpt, so we may have to check for errors. doesn't use R. refer to cole's code for the old version
    def association_rule_learning(
        self, 
        target,
        continuous=[],
        min_support=0.00045, 
        min_confidence=0.02, 
        min_items=2, 
        max_items=5, 
        min_lift=2, 
        protective=False,
        out_file='arl.csv'
    ):
        '''
        Conducts association rule learning on binarized, OHC data for given target variable.
        Saves results as a .csv, alongside with plots

        Arguments:
            target -- target column on the dataframe
            continuous -- list of continuous columns on dataframe. these will be dropped
            min_support -- minimum value of support for the rule to be considered
            min_confidence -- minimum value of confidence for the rule to be considered
            min_items -- minimum number of item in the rules including both sides
            max_items -- maximum number of item in the rules including both sides
            min_lift -- minimum value for lift for the rule to be considered
            protective -- if True, the rhs values will be flipped to find protective features
            out_file -- name of output .txt file (no extension required)
        '''
        data = self.binOHCdata_withBase.copy(deep=True)

        data.drop(columns = continuous, inplace=True)
        # remove all columns with more than 2 unique values -- this ensures all features are binarized
        for col in data.columns:
            if len(data[col].unique()) > 2:
                data.drop([col], axis=1, inplace=True)
            
        result_folder = os.path.join(self.out_folder, 'association_rule_learning')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            
        if protective:
            data[target] = np.absolute(np.array(data[target].values) - 1)

        # Generate association rules using mlxtend
        if protective:
            rules = apriori(data, min_support=min_support, use_colnames=True)
            rules = association_rules(rules, metric="lift", min_threshold=min_lift)
        else:
            rules = apriori(data, min_support=min_support, use_colnames=True)
            rules = association_rules(rules, metric="lift", min_threshold=min_lift)
            
        rules = rules[rules['consequents'].astype(str).str.contains(target)]
        # Filter rules based on length
        rules = rules[(rules['antecedents'].apply(lambda x: len(x)) >= min_items) &
                    (rules['antecedents'].apply(lambda x: len(x)) <= max_items)]

        rules = rules[rules["consequents"].astype(str).str.contains(target)]
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
        rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")

        # Sort rules by lift
        rules = rules.sort_values(by='lift', ascending=False)

        if rules.empty:
            print('No association rules found.')

        rules.to_csv(os.path.join(self.out_folder, 'association_rule_learning', 'arl_results{}.csv'.format('_protective' if protective else '')))
    
    def multivariate_linear_regression(self, target, out_file = 'mvlin.csv'):
        '''
        Runs multivariate regression on continuous, OHC data for given target variable.
        Saves results as a .csv file with given name
        '''
        y_train = self.nonbinOHCdata[target]
        x_train_sm = sm.add_constant(self.nonbinOHCdata.drop(columns = [target, 'ID_1']))

        logm2 = sm.GLM(y_train, x_train_sm, family=sm.families.Binomial())
        res = logm2.fit()
        self.display_regression_results(x_train_sm.columns, res, out_file)
    
    def multivariate_logistic_regression(self, target, out_file = 'mvlog.csv'):
        '''
        Runs multivariate regression on binarized, OHC data for given target variable.
        Saves results as a .csv file with given name
        '''
        y_train = self.binOHCdata[target]
        x_train_sm = sm.add_constant(self.binOHCdata.drop(columns = [target, 'ID_1']))

        logm2 = sm.GLM(y_train, x_train_sm, family=sm.families.Binomial())
        res = logm2.fit()
        self.display_regression_results(x_train_sm.columns, res, out_file)

    def display_regression_results(self, variable, res, out_file):
        '''
        Prints regression results and outputs them as a .csv file

        Arguments:
            variable -- list of variables used in regression
            res -- regression results
            result_file -- where the results will be outputted
        '''
        print(res.summary())
        df = pd.DataFrame(variable, columns=["variable"])
        df = pd.merge(
            df,
            pd.DataFrame(res.params, columns=["coefficient"]),
            left_on="variable",
            right_index=True,
        )
        conf_int = pd.DataFrame(res.conf_int())
        conf_int = conf_int.rename({0: "2.5%", 1: "97.5%"}, axis=1)
        df = pd.merge(df, conf_int, left_on="variable", right_index=True)
        df = pd.merge(
            df,
            pd.DataFrame(res.bse, columns=["std error"]),
            left_on="variable",
            right_index=True,
        )
        df = pd.merge(
            df,
            pd.DataFrame(res.pvalues, columns=["pvalues"]),
            left_on="variable",
            right_index=True,
        )
        adjusted = fdr_correction(res.pvalues, alpha=0.05, method="indep")[1]
        df["adjusted pval"] = adjusted
        df = df.sort_values(by="adjusted pval")
        
        result_folder = os.path.join(self.out_folder, 'regression_results')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        df.to_csv(os.path.join(result_folder, out_file))

        print("AIC: {}".format(res.aic))
        print("BIC: {}".format(res.bic))
    
# Testing the code, doesn't work rn, have to update
def main():

    # getting and formatting a test file for log regression
    data = pd.read_csv("/home/mminbay/summer_research/summer23_aylab/data/dom_overall/analysis_data.csv", index_col = 0).dropna().drop(columns = ['Age', 'Sex'])
    bin_nonOHC = data.drop(columns = ['PHQ9'])
    cont_nonOHC = data.drop(columns = ['PHQ9_binary'])
    bin_OHC_dropbase = pd.get_dummies(bin_nonOHC, columns = ['Sleeplessness/Insomnia', 'Chronotype'], drop_first = True)
    bin_OHC_withbase = pd.get_dummies(bin_nonOHC, columns = ['Sleeplessness/Insomnia', 'Chronotype'], drop_first = False)
    cont_OHC = pd.get_dummies(cont_nonOHC, columns = ['Sleeplessness/Insomnia', 'Chronotype'])

    out_folder = '/home/mminbay/summer_research/summer23_aylab/data/dom_overall/analysis/'
    r_path = '/home/mminbay/summer_research/summer23_aylab/Rscripts'
    
    # Create an instance of Stat_Analyzer
    analyzer = Stat_Analyzer(
        bin_OHC_dropbase,  
        bin_OHC_withbase,
        cont_OHC,
        bin_nonOHC, 
        cont_nonOHC, 
        r_path, 
        out_folder
    )

    # Run multivariate logistic regression
    # target_variable = 'PHQ9_binary'
    # output_file = 'test_log_reg.csv'
    # analyzer.multivariate_logistic_regression(target_variable, output_file)

    # Run multivariate linear regression
    # target_variable = 'PHQ9'
    # output_file = "test_lin_reg.csv"
    # analyzer.multivariate_linear_regression(target_variable, output_file)

    # Run association rule learning. We need the baseline for OHC as well, which is why ARL uses the df where we do not drop the baselines: binOHCdata_withBase
    # not working because we do not have arulesViz package...
    # target_variable = "PHQ9_binary"
    # output_file = "test_ARL"
    # rules = analyzer.association_rule_learning(target_variable, out_file=output_file, continuous = ['TSDI'], protective = True)

    analyzer.mediation_analysis('PHQ9', 'Chronotype_1.0', [col for col in data.columns.tolist() if col not in ['Age', 'Sex', 'ID_1', 'Chronotype_1.0', 'PHQ9_binary']], ['TSDI'])
    
    # mediation also needs R...
    
# Execute the main function
if __name__ == "__main__":
    main()
