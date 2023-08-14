import bioinfokit.analys
import itertools
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mne.stats import fdr_correction
# import multiprocessing as mp
from pathos.multiprocessing import Pool
import networkx as nx
import numpy as np
import os
import pandas as pd
import subprocess
import random
import researchpy as rp
# from rpy2.robjects.packages import importr
# from rpy2.robjects import r, pandas2ri, numpy2ri
# import rpy2.robjects as ro
# numpy2ri.activate()
# from rpy2.robjects.conversion import localconverter
from scipy.stats import sem
from scipy import stats
import scikit_posthocs as sp
import seaborn as sns
from shared_objects import SharedNumpyArray, SharedPandasDataFrame
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.genmod.families.links as links
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.genmod import families
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from utils import reorder_cols
from utilities import relabel  # hardcoded. relabel functional will be updated when anova needed
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)


'''
This class is meant to take care of all your statistical analysis. 
An example usage can be found at the end of this file
'''

class Stat_Analyzer():
    # TODO: implement!
    def __init__(
        self,
        data,
        bin_outcome,
        cont_outcome,
        out_folder,
        ohe_columns = [],
        normalize_columns = [],
        r_dir = None,        
    ):
        '''
        Arguments:
            data (DataFrame) -- dataframe to be analyzed. this should only contain the ID_1 columns, feature columns, and outcome columns. features can be normalized but not one-hot encoded.
            bin_outcome (str) -- column label of the binarized outcome on the dataframe.
            cont_outcome (str) -- column label of the continuous outcome on the dataframe.
            out_folder (str) -- where this instance will output results
            ohe_columns (list(str)) -- column identifiers to one-hot-encode
            normalize_columns (list(str)) -- column identifiers to normalize
            r_dir (str) -- path to directory containing Rscripts for some statistical analyses    
        '''
        # normalize provided columns
        for col in normalize_columns:
            min_val = data[col].min()
            max_val = data[col].max()
            data[col] = (data[col] - min_val) / (max_val - min_val)

        # prepare columns for one-hot encoding
        for col in ohe_columns:
            data[col] = data[col].astype(int)

        # modify column names for R compatibility
        new_columns = []
        for column in data.columns:
            if data[column].nunique() <= 1:
                warnings.warn('Dataframe contains column \'{}\' which has a single unique value. Removing this column.'.format(column), UserWarning)
                data.drop(columns = [column], inplace = True)
                continue
            if column[0].isdigit():
                warnings.warn('Dataframe contains column labels starting with a number and underscore. Prepending \'SNP\' for R compatibility.', UserWarning)
                new_columns.append('SNP' + column)
            else:
                new_columns.append(column)        
        data.columns = new_columns
        data.columns = data.columns.str.replace('[ ./:]', '_', regex = True)
        
        bin = data.drop(columns = [cont_outcome])
        cont = data.drop(columns = [bin_outcome])
        bin_ohc = pd.get_dummies(bin, columns = ohe_columns, drop_first = True)
        bin_ohc_wbase = pd.get_dummies(bin, columns = ohe_columns)
        cont_ohc = pd.get_dummies(cont, columns = ohe_columns, drop_first = True)

        self.bin_outcome = bin_outcome
        self.cont_outcome = cont_outcome
        
        self.bin_ohc = bin_ohc
        self.bin_ohc_wbase = bin_ohc_wbase
        self.cont_ohc = cont_ohc
        self.bin = bin
        self.cont = cont
        self.r_dir = r_dir
        self.out_folder = out_folder

        self.ARL = dict()
        self.mediation = dict()
        self.linearRegression = dict()
        self.logisticRegression = dict()
        self.oneANOVA = dict()
        self.twoANOVA = dict()

    def scheirer_ray_hare(
        self,
        snps_file_path,
        curr_snp,
        sex_column = 'Sex',
    ):
        '''
        Runs SRH test. Docstring WIP

        Arguments:
            snps_file_path (str) --
            curr_snp ('single', 'pair') --
            sex_column (str) --
        '''
        target = self.cont_outcome
        if curr_snp not in ['single', 'pair']:
            raise Exception('Argument \'curr_snp\' must be either \'single\' or \'pair\'')
        result_folder = os.path.join(self.out_folder, 'scheirer_ray_hare/')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        data = self.cont_ohc.copy(deep = True)
        data = reorder_cols(data)
        tmp_path = os.path.join(result_folder, 'srh_tmp.csv')
        data.to_csv(tmp_path, index = False)

        command = [
            "Rscript",
            os.path.join(self.r_dir, 'srh_revised.R'),
            tmp_path, 
            snps_file_path,
            curr_snp, 
            sex_column, 
            target, 
            result_folder
        ]
        
        process = subprocess.run(command, capture_output = True, text = True)
        
        if process.returncode == 0:
            print("R script executed successfully.")
            print("Output:")
            print(process.stdout)
        else:
            print("Error occurred while running the R script.")
            print("Errors:")
            print(process.stdout)
            print(process.stderr)

        os.remove(tmp_path)
        
    def one_way_anova(
        self,
        indep,
        out_file = None,
        snp_out_file = None,
        gene_name = None,
        alleles = None,
    ):
        if out_file is None:
            out_file = '{}_one_way_anova.txt'.format(indep)

        result_folder = os.path.join(self.out_folder, 'one_way_anova')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        if snp_out_file is None:
            snp_out_file = '{}_one_way_anova.png'.format(indep)

        if 'pair' in indep:
            comps = indep.split(sep=':')
            indep = '_'.join(comps)
        
        data = self.cont_ohc.copy(deep = True)
        target = self.cont_outcome
        
        # One-way ANOVA assumptions
        # 1. Independence - Assumed to be met as each observation is independent.
        # 2. Normality - Test the normality assumption using Shapiro-Wilk test or visual inspection (e.g., Q-Q plot).
        # 3. Homogeneity of Variance - Test the homogeneity of variance using Levene's test or visual inspection (e.g., boxplot).
        
        # Check normality assumption using Shapiro-Wilk test

        f = open(os.path.join(result_folder, out_file), "w+")
        f.write("Shapiro-Wilk Test for Normality:")
        for group, group_data in data.groupby(indep):
            f.write(f"Group {group}: p-value = {stats.shapiro(group_data[target])[1]:.4f}")
        
        # Check homogeneity of variance assumption using Levene's test
        f.write("\nLevene's Test for Homogeneity of Variance: ")
        grouped_data = [group_data[target].values for _, group_data in data.groupby(indep)]
        f.write(f"p-value = {stats.levene(*grouped_data)[1]:.4f}")
        
        # Perform one-way ANOVA
        f.write("\nOne-Way ANOVA:\n")
        model = sm.formula.ols('{} ~ {}'.format(target, indep), data=data).fit()
        anova_table = sm.stats.anova_lm(model)
        f.write(anova_table.to_string())
        
        # Post hoc test (Tukey's HSD) to compare group means
        f.write("\nTukey's HSD Post Hoc Test:")
        tukey_result = sm.stats.multicomp.pairwise_tukeyhsd(data[target], data[indep])
        f.write(str(tukey_result))
        
        # Kruskal-Wallis assumptions
        # 1. Independence - Assumed to be met as each observation is independent.
        # 2. Ordinal Data - The dependent variable should be measured on an ordinal scale or higher.
        
        # Check assumptions
        # As Kruskal-Wallis is a non-parametric test, it does not assume normality or homogeneity of variance.
        
        # Perform Kruskal-Wallis test
        f.write("\nKruskal-Wallis Test:")
        kruskal_result = stats.kruskal(*[group_data[target].values for _, group_data in data.groupby(indep)])
        f.write(f"p-value = {kruskal_result.pvalue:.4f}")
        
        # Post hoc test (Dunn's test) to compare group medians
        f.write("\nDunn's Test Post Hoc Test:\n")
        dunn_result = sp.posthoc_dunn(data, val_col = target, group_col = indep)
        f.write(str(dunn_result))

        ########## ONE SNP PLOT
        # Group the data by 'SNP' and calculate the mean
        mean_vals = data.groupby([indep], as_index = False)[target].mean()
        
        # Calculate the Standard Error of the mean for each group
        sem = data.groupby([indep], as_index = False)[target].sem()
        
        # Create a new DataFrame to store the results
        grouped_data = mean_vals
        grouped_data.columns = ['SNP', 'Mean_'+target]
        grouped_data['St.Err'] = sem[target]

        if gene_name == None:
            gene_name = 'not_provided'
        if alleles == None:
            alleles = ['X', 'Y']
            
        self.one_way_snp_bar_plot(indep, gene_name, alleles[0], alleles[1], grouped_data, snp_out_file)
        
    def one_way_snp_bar_plot(self, snp_name, gene_name, normal_allele, variant_allele, df, output_file):
         # Filter the dataframe based on the normal and variant alleles
        normal_data = df[df['SNP'] == 0]
        variant_data = df[df['SNP'] == 1]
    
        # Extract the required data from the dataframes
        normal_mean = normal_data['Mean_PHQ9'].values[0]
        variant_mean = variant_data['Mean_PHQ9'].values[0]
        normal_st_err = normal_data['St.Err'].values[0]
        variant_st_err = variant_data['St.Err'].values[0]
    
        # Set up the bar plot
        bars = plt.bar([0, 1], [normal_mean, variant_mean], yerr=[normal_st_err, variant_st_err], tick_label=[normal_allele, variant_allele], color=['gray', 'lightgray'], capsize=5, edgecolor='black')
    
        # Add labels and title
        plt.xlabel('SNP')
        plt.ylabel('Mean PHQ9')
        plt.title(f'{gene_name} ({normal_allele}/{variant_allele})')

        plt.ylim(0, max(normal_mean, variant_mean) + 1)
        
        # Show the legend with both alleles
        plt.legend(bars, [normal_allele, variant_allele])
    
        # Show the plot
        plt.show()
        
        # Create the result folder if it doesn't exist
        result_folder = os.path.join(self.out_folder, 'one_way_anova_plots')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Save the plot to the specified output file
        plt.savefig(os.path.join(result_folder, output_file))
        plt.close()

    def two_way_anova(
        self,
        indep_1,
        indep_2 = 'Sex',
        out_file = None,
        snp_out_file = 'two_way_anova.png',
        snp_pair_out_file = 'snp_pair_anova.png',
        alleles = [],
        gene_names = []
    ):
        
        if out_file is None:
            out_file = '{}_{}_two_way_anova.txt'.format(indep_1, indep_2)

        result_folder = os.path.join(self.out_folder, 'two_way_anova')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

            
        # assigning alleles
        one_gene, two_genes = False, False
        gene_1, gene_2 = None, None
        if len(alleles) == 2:
            wild = alleles[0]
            mut = alleles[1]
            if len(gene_names) == 1:
                gene_N = gene_names[0]+'_'+wild+'/'+mut
                one_gene = True
        elif len(alleles) == 4:
            wild1, wild2 = alleles[0], alleles[2]
            mut1, mut2 = alleles[1], alleles[3]
            if len(gene_names) == 2:
                gene_1 = gene_names[0]+'_'+wild1+'/'+mut1
                gene_2 = gene_names[1]+'_'+wild2+'/'+mut2
                two_genes = True
                gene_couple = gene_1+'--'+gene_2
                print('here')
        elif len(alleles) != 0:
            raise Exception('Please provide a list of 2 alleles in case of 1 SNP, and 4 alleles in the case of a snp pair \n(or do not provide any if you do not want custom axes names). Allele name examples: \nsingle: [AA, GG]; pair: [AA, TT, GG, TT] (where first two are the normal and variant version \nof the snps). Since this deals with the dominant model, you can say TX where X can be T or others \nsince T, the variant, is considered dominant in this model.')
    
        data = self.cont_ohc.copy(deep = True)
        target = self.cont_outcome

        f = open(os.path.join(result_folder, out_file), "w+")
        # changing 'pair:rs###:rs#####' to the form of: 'pair_rs###_rs#####'
        if 'pair' in indep_1:
            comps = indep_1.split(sep=':')
            indep_1 = '_'.join(comps)
        # Two-way ANOVA assumptions
        # 1. Independence - Assumed to be met as each observation is independent.
        
        # Perform two-way ANOVA with Type 3
        formula = '{} ~ C({}) + C({}) + C({}):C({})'.format(target, indep_1, indep_2, indep_1, indep_2)
        model_anova = smf.ols(formula, data = data).fit()
        anova_table = sm.stats.anova_lm(model_anova, typ=3)
        
        f.write("\nType 3 Two-Way ANOVA:\n")
        f.write(anova_table.to_string())
        
        # Check normality assumption for the errors in ANOVA using residuals
        residuals_anova = model_anova.resid
        shapiro_test_p_value = stats.shapiro(residuals_anova)[1]
        f.write("\nShapiro-Wilk Test for Normality (ANOVA Residuals):")
        f.write("p-value = {}".format(str(shapiro_test_p_value)))
        
        # Check equality of variance using Levene's test
        levene_p_value = stats.levene(*[group_data[target] for _, group_data in data.groupby([indep_2, indep_1])])[1]
        f.write("\nLevene's Test for Homogeneity of Variance:")
        f.write("p-value = {}".format(str(levene_p_value)))
                
        outliers = self.__detect_outliers_tukey(data, group_col=[indep_2, indep_1], y_col = target, threshold = 1.5)
        f.write("\n{} outliers detected:".format(len(data.loc[outliers])))
        f.write(str(data.loc[outliers]['ID_1'].tolist()))
        
        # Perform Tukey's test as a follow-up for ANOVA
        f.write("\nTukey's Test as a Follow-up for ANOVA:")
        mc = MultiComparison(data[target], data[indep_1])
        result = mc.tukeyhsd()
        f.write(str(result))
      
        ########## ONE SNP PLOT
        # Group the data by 'SNP' and 'Sex' and calculate the mean for each group
        mean_vals = data.groupby([indep_2, indep_1], as_index = False)[target].mean()
        # Calculate the Standard Error of the mean for each group
        sem = data.groupby([indep_2, indep_1], as_index = False)[target].sem()
        
        # Create a new DataFrame to store the results
        grouped_data = mean_vals
        grouped_data.columns = [indep_2, 'SNP', 'Mean_'+target]
        grouped_data['St.Err'] = sem[target]

        grouped_data[indep_2] = grouped_data[indep_2].replace({1: 'Male', 0: 'Female'})
        self.one_snp_bar_plot(grouped_data, snp_out_file, gene_names, alleles)
                              
        ########## SNP PAIR PLOT     
        # # if a snp pair is provided, make this plot in addition to the one above
        # if 'pair' in indep_1:
        #     snp_pair = indep_1.split(sep='_')
        #     snpx = snp_pair[1]
        #     snpy = snp_pair[2]
            
        #     # Group the data by 'sex' and 'SNP' and calculate the mean for each group
        #     mean_vals = data.groupby([indep_2, snpx, snpy], as_index = False)[target].mean()
        #     # Calculate the Standard Error of the mean for each group
        #     sem = data.groupby([indep_2, snpx, snpy], as_index = False)[target].sem()

        #     # Create a new DataFrame to store the results
        #     grouped_data = mean_vals
        #     grouped_data.columns = [indep_2, 'SNPX', 'SNPY', 'Mean_'+target]

        #     grouped_data['St.Err'] = sem[target]
            
        #     grouped_data[indep_2] = grouped_data[indep_2].replace({1: 'Male', 0: 'Female'})
        #     two_genes = False
        #     if two_genes:
        #         grouped_data.columns = [indep_2, gene_1, gene_2, 'Mean_'+target, 'St.Err']
        #     else:
        #         grouped_data.columns = [indep_2, 'SNPX', 'SNPY', 'Mean_'+target, 'St.Err']
        #     print(grouped_data)
        #     self.anova_pair_snp_plot(grouped_data, snp_pair_out_file, snpx=gene_1, snpy=gene_2)

            
    def one_snp_bar_plot(self, data, out_file, gene_names=None, alleles=None):
        # Sample data in a pandas DataFrame
        '''
        data = pd.DataFrame({
            "Sex": ["Male", "Male", "Female", "Female"],
            "SNP": [0, 1, 0, 1],
            "Mean_PHQ9": [10, 15, 20, 12],
            "St.Err": [1, 2, 1.5, 1]
        })
        '''
       
        # Filter data for SNP 0 and 1
        snp_0_data = data[data['SNP'] == 0]
        snp_1_data = data[data['SNP'] == 1]
        # Define the positions for the bars
        bar_width = 0.35
        bar_positions_snp0 = [0, 1]
        bar_positions_snp1 = [pos + bar_width for pos in bar_positions_snp0]
        
        # Plot barplot for SNP 0
        plt.bar(bar_positions_snp0, snp_0_data['Mean_PHQ9'], width=bar_width, yerr=snp_0_data['St.Err'], label='WT', color='white', edgecolor='black', capsize=5)
        # Plot barplot for SNP 1
        plt.bar(bar_positions_snp1, snp_1_data['Mean_PHQ9'], width=bar_width, yerr=snp_1_data['St.Err'], label='Variant', color='darkgray', edgecolor='black', capsize=5)
        
        # Calculate the maximum of the bar heights
        max_bar_height = max(data['Mean_PHQ9'] + data['St.Err']) + 0.5
        
        # Set the y-axis limit to 1 plus the maximum bar height
        plt.ylim(0, max_bar_height)
        
        # Set the x-axis ticks and labels
        plt.xticks([pos + bar_width/2 for pos in bar_positions_snp0], ['Male', 'Female'])
        
        # Add labels and title
        plt.xlabel('Sex')
        plt.ylabel('Mean PHQ9')
        if gene_names != None:
            genes = '-'.join(gene_names)
        else:
            genes = 'gene'
        plt.title(f'Mean PHQ9 by Sex and SNP in {genes}')
        
        # Add legend
        plt.legend()
        
        # Show the plot
        plt.show()
        
        result_folder = os.path.join(self.out_folder, 'two_way_anova_plots')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
        plt.savefig(os.path.join(result_folder, out_file))  # add: , dpi=300 to save with 300 dpi resolution
        plt.close()

    def anova_pair_snp_plot(self, data, out_file, snpx=None, snpy=None, alleles=None):

        '''
        # Sample data in a pandas DataFrame
        data = pd.DataFrame({
            "Sex": ["Male", "Male", "Male", "Male","Female", "Female", "Female", "Female"],
            "SNPX": ["Type C", "Type C", "Type D", "Type D","Type C", "Type C", "Type D", "Type D"],
            "SNPY": ["Type A", "Type B", "Type A", "Type B","Type A", "Type B", "Type A", "Type B"],
            "Mean_PHQ9": [10, 15, 20, 12, 11, 13, 24, 21],
            "St.Err": [1, 2, 1.5, 1, 3, 4, 3, 2]
        })
        '''
        # chat gpt's code not working

        # result_folder = os.path.join(self.out_folder, 'two_way_anova_plots')
        # if not os.path.exists(result_folder):
        #     os.makedirs(result_folder)
        # plt.savefig(os.path.join(result_folder, out_file))  # add: , dpi=300 to save with 300 dpi resolution
    
    def __detect_outliers_tukey(self, data, group_col, y_col, threshold=1.5):
        outliers = []
        for group, group_data in data.groupby(group_col):
            q1 = group_data[y_col].quantile(0.25)
            q3 = group_data[y_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers_group = group_data[(group_data[y_col] < lower_bound) | (group_data[y_col] > upper_bound)]
            outliers.extend(outliers_group.index.tolist())
        return outliers

    def mediation_analysis(
        self,
        dep,
        med,
        indep,
        mediator_type = 'bin',
        sims = 100,
        covariates = None
    ):
        '''
        Conducts mediation analysis on continous, OHC data for given dependent variable, mediator, and independent variable(s).
        Saves results as a .csv file

        Arguments:
            dep ('bin', 'cont') -- which outcome variable to use 
            med (str) -- column identifier of mediator variable
            indep (str or list(str)) -- (list of) column identifier of independent variable(s)
            mediatior_type ('bin', 'cont') -- whether the mediator variable is binary or continuous. 
            sims (int) -- how many simulations to run for each mediation analysis
            covariates (list(str)) -- covariates to include in mediation analysis. pass 'ALL' to include every variable other than target, med, and indep
        '''
        if dep == 'bin':
            target = self.bin_outcome
            data = self.bin_ohc.copy(deep = True)
        elif dep == 'cont':
            target = self.cont_outcome
            data = self.cont_ohc.copy(deep = True)
        else:
            raise Exception('Argument \'dep\' for mediation analysis must be either \'bin\' or \'cont\'')

        data.drop(columns = ['ID_1'], inplace = True)

        if covariates == 'ALL':
            covariates = data.drop(columns = [med, target]).columns.tolist()
        elif covariates is not None:
            for i in range(len(covariates)):
                covariates[i] = covariates[i].replace(' ', '_').replace('.', '_').replace('/', '_').replace(':', '_')
                if '+' in covariates[i]:
                    raise Exception('Forbidden character \'+\' in column identifier for covariate {}'.format(covariates[i]))
            covariates = data.filter(regex = '|'.join(covariates)).columns.tolist() # in case the covariate is OHE
        else:
            covariates = 'DNE'
            covariates_str = 'DNE'
        
        med = med.replace('.', '_').replace(' ', '_').replace('/', '_').replace(':', '_')
        target = target.replace('.', '_').replace(' ', '_').replace('/', '_').replace(':', '_')

        result_folder = os.path.join(self.out_folder, 'mediation_analysis')

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        tmp_path = os.path.join(result_folder, 'ma_tmp.csv')
        data.to_csv(tmp_path)
        
        # convert indep to a list if a single var was passed
        if type(indep) == str:
            t = list(); t.append(indep)
            indep = t
        
        for i in range(len(indep)):
            var = indep[i].replace('.', '_').replace(' ', '_').replace('/', '_').replace(':', '_')
            if covariates != 'DNE':
                if var in covariates:
                    covariates.remove(var)
                covariates_str = '+'.join(covariates)
            print(var, covariates_str)
            command = [
                "Rscript",
                os.path.join(self.r_dir, 'mediationAnalysis.R'),
                tmp_path, 
                var,
                med,
                mediator_type,
                target, 
                dep,
                str(sims),
                covariates_str
            ]

            covariates.append(var)
            
            process = subprocess.run(command, capture_output = True, text = True)
            
            if process.returncode == 0:
                print("R script executed successfully.")
                print("Output:")
                print(process.stdout)
            else:
                print("Error occurred while running the R script.")
                print("Errors:")
                print(process.stdout)
                print(process.stderr)
    
        os.remove(tmp_path)

    def association_rule_learning(
        self, 
        continuous = [],
        min_support = 0.00045, 
        min_confidence = 0.02, 
        min_items = 2, 
        max_items = 5, 
        min_lift = 2, 
        protective = False,
        drop_pairs = True,
        drop_clinical = False,
        out_file = 'arl.txt'
    ):
        '''
        Conducts association rule learning on binarized, OHC data for given target variable.
        Saves results as a .csv, alongside with plots

        Arguments:
            continuous (list(str)) -- list of continuous column identifiers on dataframe. these will be dropped
            min_support (float) -- minimum value of support for the rule to be considered
            min_confidence (float) -- minimum value of confidence for the rule to be considered
            min_items (int) -- minimum number of item in the rules including both sides
            max_items (int) -- maximum number of item in the rules including both sides
            min_lift (float) -- minimum value for lift for the rule to be considered
            protective (bool) -- if True, the target value will be flipped to find protective features
            drop_pairs (bool) -- if True, columns containing 'pair:' will be dropped
            drop_clinical (bool) -- if True, columns that are not SNPs or pairs will be dropped
            out_file (str) -- name of output file. requires .txt extension
        '''
        data = self.bin_ohc_wbase.copy(deep = True)
        print(data.columns)
        target = self.bin_outcome
        # the following column modifications are hot garbage, please make them more efficient
        
        if drop_pairs:
            data.drop(columns = data.filter(regex = '^pair:').columns, inplace = True)
        if drop_clinical:
            data = data.filter(regex = '^(rs|SNP|pair|' + target + ')')

        data.drop(columns = continuous, inplace = True)
        # remove all columns with more than 2 unique values -- this ensures all features are binarized
        for col in data.columns:
            if len(data[col].unique()) > 2:
                data.drop([col], axis=1, inplace=True)

        data = data.astype(int)        
        if protective:
            data[target] = ~data[target]
        data = reorder_cols(data)
        result_folder = os.path.join(self.out_folder, 'association_rule_learning')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        data.to_csv(os.path.join(result_folder, 'AprioriData.csv'), index = False)
        csv_path = os.path.join(result_folder, out_file.split('.')[0] + '.csv')

        command = [
            "Rscript",   
            os.path.join(self.r_dir, 'Association_Rules.R'),
            result_folder, 
            str(min_support), 
            str(min_confidence), 
            str(max_items), 
            str(min_items), 
            target, 
            str(min_lift),
            csv_path
        ]
        
        process = subprocess.run(command, capture_output = True, text = True)

        if process.returncode == 0:
            print("R script executed successfully.")
            print("Output:")
            print(process.stdout)
        else:
            print("Error occurred while running the R script.")
            print("Errors:")
            print(process.stdout)
            print(process.stderr)
            raise Exception('za')

        os.remove(os.path.join(result_folder, 'AprioriData.csv'))
        if os.path.exists(csv_path):
            ARLRules = pd.read_csv(csv_path)
            pvals = ARLRules['pValue']
            pvals = fdr_correction(pvals, alpha=0.05, method='indep')
            ARLRules['adj pVals'] = pvals[1]
        else:
            print('No rules meeting minimum requirements were found')
            print('Process Terminated')
            return

        vars = ARLRules['LHS'].tolist()
        features, newF, rows, pvals = list(), list(), list(), list()
        oddsRatios = pd.DataFrame(columns=['LHS-RHS', 'Odds Ratio', 'Confidence Interval', 'pValue', 'adjusted pVal'])
        for var in vars:
            newF.append(var)
            features.append(var.replace('{', '').replace('}', '').split(','))

        shared_data = SharedPandasDataFrame(data)

        args = [(features[i], newF[i], shared_data) for i in range(len(features))]

        def arl_stats_wrapper(cols, newFeature, shared_data_pointer):
            dataC = shared_data_pointer.read()[cols]
            dataC[newFeature] = dataC[dataC.columns[:]].apply(lambda x: ','.join(x.astype(str)),axis=1)
            dataC = dataC[[newFeature]]
            dataC[target] = shared_data_pointer.read()[target]
            toDrop = []
            for index, r in dataC.iterrows():                
                fValue = set(r[newFeature].split(','))
                if (len(fValue) > 1):
                    toDrop.append(index)
            dataC.drop(toDrop, inplace = True)
            dataTrue = dataC[dataC[target] == 1].drop([target], axis =1).value_counts().tolist()
            dataFalse = dataC[dataC[target] == 0].drop([target], axis = 1).value_counts().tolist()
            if len(dataTrue) == 1:
                dataTrue.append(0)
            if len(dataFalse) == 1:
                dataFalse.append(0)
            dataTrue.reverse(); dataFalse.reverse()
            
            table = np.array([dataTrue, dataFalse])
            print(table)
            res = statsmodels.stats.contingency_tables.Table2x2(table, shift_zeros = True)

            return [str(newFeature)+'-'+str(target), res.oddsratio, res.oddsratio_confint(), res.oddsratio_pvalue()]

        with Pool() as pool:
            rows = pool.starmap(arl_stats_wrapper, args)

        shared_data.unlink()
        pvals = [row[-1] for row in rows]           
        pvals = fdr_correction(pvals, alpha=0.05, method='indep')
        for i in range(len(pvals[1])):
            rows[i].append(pvals[1][i])
            oddsRatios.loc[len(oddsRatios.index)] = rows[i]
                      
        Association = open(os.path.join(result_folder, out_file), 'w')
        Association.write('\n\n----------------------------------------------------------------------------------------\n\n')
        Association.write('\n\nOdds Ratio analysis for Association Rule Learning: \n----------------------------------------------------------------------------------------\n\n')
        for i in range(len(oddsRatios)):
            two_var = oddsRatios.iloc[i, :]
            two_var = two_var.to_frame()
            variables = str(two_var.iloc[0,0]).split('-')
            two_var = two_var.iloc[1: , :]
            Association.write('The odds ratio, p-Value, and confidence interval between ' + variables[0]+' and ' + variables[1] + ' are: \n\n')
            toWrite = two_var.to_string(header = False, index = True)
            Association.write(toWrite+'\n')
            Association.write('----------------------------------------------------------------------------------------\n\n')

        Association.close()
        
    def multivariate_linear_regression(self, out_file = 'mvlin.csv'):
        '''
        Runs multivariate regression on continuous, OHC data for given target variable.
        Saves results as a .csv file with given name
        '''
        target = self.cont_outcome
        y_train = self.cont_ohc[target]
        x_train_sm = sm.add_constant(self.cont_ohc.drop(columns = [target, 'ID_1']))

        logm2 = sm.GLM(y_train, x_train_sm, family=sm.families.Gaussian())
        res = logm2.fit()
        self.display_regression_results(x_train_sm.columns, res, out_file)
    
    def multivariate_logistic_regression(self, out_file = 'mvlog.csv'):
        '''
        Runs multivariate regression on binarized, OHC data for given target variable.
        Saves results as a .csv file with given name
        '''
        target = self.bin_outcome
        y = self.bin_ohc[target]
        X = self.bin_ohc.drop(columns = ['ID_1'])

        res=sm.GLM(y, X.loc[:, X.columns != 'PHQ9_binary'], family = sm.families.Binomial(link = sm.families.links.logit())).fit()

        # res = sm.GLM(y, X, family = sm.families.Binomial(link = sm.families.links.logit())).fit()
        self.display_regression_results(X.columns, res, out_file, calc_odds = True)

    def display_regression_results(self, variable, res, out_file, calc_odds = False):
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
        if calc_odds:
            df['odds_ratio'] = np.exp(df['coefficient'])
        df = df.sort_values(by="adjusted pval")
        
        result_folder = os.path.join(self.out_folder, 'regression_results')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        df.to_csv(os.path.join(result_folder, out_file))

        print("AIC: {}".format(res.aic))
        print("BIC: {}".format(res.bic))
    
# Testing the code, doesn't work rn, have to update
def main():
    pass
    
# Execute the main function
if __name__ == "__main__":
    main()
