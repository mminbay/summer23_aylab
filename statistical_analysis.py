import os
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
import pandas as pd
import statsmodels.api as sm
import bioinfokit.analys
from mne.stats import fdr_correction
import researchpy as rp
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri
numpy2ri.activate()
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import itertools
import statsmodels
from utilities import relabel
from data_processor import data_processor
from scipy.stats import sem
from scipy import stats
import random

'''
This class is meant to take care of all your statistical analysis. 
An example usage can be found at the end of this file
'''

class Stat_Analyzer():
    # TODO: implement!
    def __init__(
        self,
        binOHCdata,
        nonbinOHCdata,
        binnonOHCdata,
        nonbinnonOHCdata,
        r_path,
        out_folder
    ):
        '''
        Arguments:
            binOHCdata -- dataframe to be analyzed, where categorical features are OHC'd and outcome is binarized
            nonbinOHCdata -- dataframe to be analyzed, where categorical features are OHC'd and outcome is continuous
            binnonOHCdata -- dataframe to be analyzed, where categorical features are not OHC'd and outcome is binarized
            nonbinnonOHCdata -- dataframe to be analyzed, where categorical features are not OHC'd and outcome is continuous
            r_path -- path to directory containing Rscripts that are needed for some statistical analysis
            out_folder -- where this instance will output results

        '''
        self.binOHCdata = binOHCdata
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
            current_var = str(var) + '-' + str(med) + '-' + str(dep) + '.txt'
            result_file = os.path.join(result_folder, current_var)

            l1 = importr('mediation')
            formulaMake = r['as.formula']
            mediate, lm, glm, summary, capture = r['mediate'], r['lm'], r['glm'], r['summary'], r['capture.output']

            MediationFormula = formulaMake(mediator + ' ~ ' + var)
            OutcomeFormula = formulaMake(dep + ' ~ ' + var + ' + ' + mediator)

            with localconverter(ro.default_converter + pandas2ri.converter):
                data = ro.conversion.py2rpy(data)

            if med in continuous:
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
            capture(dfR, file = result_file)

    def association_rule_learning(
        self, 
        target,
        continuous = [],
        min_support = 0.00045, 
        min_confidence = 0.02, 
        min_items = 2, 
        max_items = 5, 
        min_lift = 2, 
        protective = False,
        out_file = 'arl.csv'
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
        data = self.binOHCdata.copy(deep = True)
        data.columns = data.columns.str.replace(' ', '_')
        data.columns = data.columns.str.replace('.', '_')

        # remove all columns with more than 2 unique values -- this ensures all features are binarized
        for col in data.columns:
            if len(data[col].unique()) > 2:
                data.drop([col], axis = 1, inplace = True)
        
        data.drop(continuous, axis = 1, inplace = True)
        
        result_folder = os.path.join(self.out_folder, 'association_rule_learning')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        if protective:
            data[target] = np.absolute(np.array(data[target].values) - 1)
        
        apriori_path = os.path.join(result_folder, out_file + '_apriori.csv')
        data.to_csv(apriori_path) # tmp file for Rscript to work with
        args = result_folder + ' ' + str(min_support) + ' '  + str(min_confidence) + ' ' + str(max_items) + ' ' + str(min_items) + ' ' + str(target) + ' ' + str(min_lift)
        os.system('Rscript ' + os.path.join(self.r_path, 'Association_Rules.R') + ' ' + args) # execute R from cmd
        os.remove(apriori_path) # remove the tmp file

        r_result_path = os.path.join(result_folder, 'apriori.csv')
        if os.path.exists(r_result_path):
            ARLRules = pd.read_csv(r_result_path)
            pvals = ARLRules['pValue']
            pvals = fdr_correction(pvals, alpha=0.05, method='indep')
            ARLRules['adj pVals'] = pvals[1]
        else:
            print('No rules meeting minimum requirements were found')
            print('Process Terminated')
            return
        os.remove(r_result_path)

        variables = ARLRules['LHS'].tolist()
        features, newF, rows, pvals = list(), list(), list(), list()
        oddsRatios = pd.DataFrame(columns=['LHS-RHS', 'Odds Ratio', 'Confidence Interval', 'pValue', 'adjusted pVal'])
        for var in variables:
            newF.append(var)
            features.append(var.replace('{', '').replace('}', '').split(','))
        for i in range(len(features)):
            cols = features[i]
            newFeature = newF[i]
            dataC = data.drop([x for x in data.columns if x not in cols], axis = 1)
            dataC[newFeature] = dataC[dataC.columns[:]].apply(lambda x: ','.join(x.astype(str)),axis=1)
            dataC = dataC[[newFeature]]
            dataC[target] = data[target]
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
            
        association = open(os.path.join(result_folder, out_file + '.txt'), 'w')
        association.write(ARLRules.to_string(index=False))
        association.write('\n\n----------------------------------------------------------------------------------------\n\n')
        association.write('\n\nOdds Ratio analysis for Association Rule Learning: \n----------------------------------------------------------------------------------------\n\n')
        for i in range(len(oddsRatios)):
            two_var = oddsRatios.iloc[i, :]
            two_var = two_var.to_frame()
            variables = str(two_var.iloc[0,0]).split('-')
            two_var = two_var.iloc[1: , :]
            association.write('The odds ratio, p-Value, and confidence interval between ' +variables[0]+' and ' + variables[1] + ' are: \n\n')
            toWrite = two_var.to_string(header = False, index = True)
            association.write(toWrite+'\n')
            association.write('----------------------------------------------------------------------------------------\n\n')
        association.close()

        if protective:
            self.ARL['ARLProtect'] = ARLRules
        else:
            self.ARL['ARLRisk'] = ARLRules
        
        return ARLRules
    
    def multivariate_linear_regression(self, target, out_file = 'mvlin.csv'):
        '''
        Runs multivariate regression on continuous, OHC data for given target variable.
        Saves results as a .csv file with given name
        '''
        y_train = self.nonbinOHCdata[target]
        x_train_sm = sm.add_constant(self.nonbinOHCdata.drop(columns = [target]))

        logm2 = sm.GLM(y_train, x_train_sm, family=sm.families.Binomial())
        res = logm2.fit()
        display_regression_results(x_train_sm.columns, res, out_file)
    
    def multivariate_logistic_regression(self, target, out_file = 'mvlog.csv'):
        '''
        Runs multivariate regression on binarized, OHC data for given target variable.
        Saves results as a .csv file with given name
        '''
        y_train = self.binOHCdata[target]
        x_train_sm = sm.add_constant(self.binOHCdata.drop(columns = [target]))

        logm2 = sm.GLM(y_train, x_train_sm, family=sm.families.Binomial())
        res = logm2.fit()
        display_regression_results(x_train_sm.columns, res, out_file)

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
        adjusted = fdrcorrection(res.pvalues, method="indep", is_sorted=False)[1]
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
    clinical_df = pd.read_csv("/datalake/AyLab/depression_testing/depression_data_ohc.csv", index_col = 0)
    clinical_df.drop(['PHQ9', 'Sex', 'Age'], axis=1, inplace=True)
    snp_df = pd.read_csv("/datalake/AyLab/depression_snp_data/dominant_model/dominant_depression_allsnps_6000extra_c1.csv", index_col = 0, nrows = 1000)
    snp_df.drop(['PHQ9_binary', 'Sex'], axis=1, inplace=True)
    all_cols = snp_df.columns.tolist()
    snp_cols = [col for col in all_cols if col not in ['ID_1']]
    random_50_snps_withID = random.sample(snp_cols, 50) # randomly taking 50 snps for testing
    random_50_snps_withID.append('ID_1') # adding ID to the snps
    snp_df = snp_df[random_50_snps_withID]
    
    snp_and_clinical_df = pd.merge(snp_df, clinical_df, on='ID_1')
    snp_and_clinical_df.dropna(inplace = True) # drop rows with missing values

# init values
    binOHCdata = snp_and_clinical_df 
    out_folder = '/home/akhan/repo_punks_mete/summer23_aylab/data/'
    nonbinOHCdata, binnonOHCdata, nonbinnonOHCdata, r_path= None
    
# Create an instance of Stat_Analyzer
    analyzer = Stat_Analyzer(
        binOHCdata, 
        nonbinOHCdata, 
        binnonOHCdata, 
        nonbinnonOHCdata, 
        r_path, 
        out_folder
    )

    
    
    # Run multivariate logistic regression
    target_variable = "PHQ9_Binary"
    output_file = "test_multvar_log_reg.csv"
    analyzer.multivariate_logistic_regression(target_variable, output_file)

# Execute the main function
if __name__ == "__main__":
    main()
