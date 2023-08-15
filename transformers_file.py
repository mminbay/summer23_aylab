import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from statistical_analysis import Stat_Analyzer

# excel_file_path = '/home/mminbay/summer_research/med_an/depression_study_regressions_sig_factors.xlsx'
# sexes = ['MALE', 'FEMALE', 'OVERALL']
# regs = ['LIN', 'LOG']

# for sex in sexes:
#     for reg in regs:
#         sheet_name = 'SNPS {} {}'.format(sex, reg)
#         data = pd.read_excel(excel_file_path, sheet_name=sheet_name)
#         csv_file_path = '/home/mminbay/summer_research/med_an/snps_{}_{}.csv'.format(sex.lower(), reg.lower())  
#         data.to_csv(csv_file_path, index = False)

sexes = ['male', 'female', 'overall']
regs = ['log', 'lin']
snps_format = '/home/mminbay/summer_research/final_runs/snps_{}_{}.csv'
data_format = '/home/mminbay/summer_research/final_runs/{}_final_features_2g.csv'

for sex in sexes:
    data = pd.read_csv(data_format.format(sex), index_col = 0)
    factors_lin = pd.read_csv(snps_format.format(sex, 'lin'))
    factors_lin.columns = factors_lin.columns.str.strip()
    
    factors_log = pd.read_csv(snps_format.format(sex, 'log'))
    factors_log.columns = factors_log.columns.str.strip()
    
    factors = pd.concat([factors_lin, factors_log])
    factors['variable'] = factors['variable'].str.strip()
    factors.drop_duplicates(subset = 'variable')
    factors_list = factors['variable'].tolist()
    factors_list.extend(['Sex', 'Age_n', 'TSDI_n', 'Sleeplessness/Insomnia', 'Chronotype', 'Overall_Health_Score', 'ID_1', 'PHQ9_binary', 'PHQ9'])

    data_final = data[factors_list].copy(deep = True)

    sa = Stat_Analyzer(
        data_final,
        'PHQ9_binary',
        'PHQ9',
        '/home/mminbay/summer_research/final_runs/{}_clinical_NEW'.format(sex),
        ohe_columns = ['Overall_Health_Score'],
        r_dir = '/home/mminbay/summer_research/summer23_aylab/Rscripts'
    )

    sa.association_rule_learning()

# train_data = pd.read_csv('/home/mminbay/summer_research/depression_study/{}_train_final_features.csv'.format(sex), index_col = 0).drop(columns = ['PHQ9'])
# train_data['Chronotype_bin'] = (train_data['Chronotype'] > 2).astype(int)
# train_data.drop(columns = ['Chronotype'], inplace = True)
# train_data['Sleeplessness/Insomnia_bin'] = (train_data['Sleeplessness/Insomnia'] > 2).astype(int)
# train_data.drop(columns = ['Sleeplessness/Insomnia'], inplace = True)
# test_data = pd.read_csv('/home/mminbay/summer_research/depression_study/{}_test_final_features.csv'.format(sex), index_col = 0).drop(columns = ['PHQ9'])
# test_data['Chronotype_bin'] = (test_data['Chronotype'] > 2).astype(int)
# test_data.drop(columns = ['Chronotype'], inplace = True)
# test_data['Sleeplessness/Insomnia_bin'] = (test_data['Sleeplessness/Insomnia'] > 2).astype(int)
# test_data.drop(columns = ['Sleeplessness/Insomnia'], inplace = True)

# prepare feature matrix and outcome vector
# X_train = train_data.drop(columns = ['ID_1', target_column]).to_numpy()
# y_train = train_data[target_column].to_numpy()

# X_test = test_data.drop(columns = ['ID_1', target_column]).to_numpy()
# y_test = test_data[target_column].to_numpy()

# classifier = XGBClassifier(
#     booster = 'gbtree',
#     max_depth = 13,
#     eta = 0.005324816318391432,
#     min_child_weight = 100, 
#     subsample = 0.13198248894566342,
#     colsample_bytree = 0.6605161721134994,
#     n_estimators = 500, 
#     use_label_encoder = False, 
#     scale_pos_weight = 61
# )

# classifier.fit(X_train, y_train)

# display = PrecisionRecallDisplay.from_estimator(
#     classifier, X_test, y_test, name="XGBoost", plot_chance_level=True
# )

# _ = display.ax_.set_title("Precision-Recall curve")

# plt.savefig("precision_recall_curve.png")

# plt.show()

# print('joe')

