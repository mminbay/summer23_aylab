import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from statistical_analysis import Stat_Analyzer

sex = 'overall'
target_column = 'PHQ9_binary'

data = pd.read_csv('/home/mminbay/jonathan/overall_final_features_2g.csv', index_col = 0)

sa = Stat_Analyzer(
    data,
    'PHQ9_binary',
    'PHQ9',
    '/home/mminbay/jonathan/',
    ohe_columns = ['Chronotype', 'Sleeplessness/Insomnia', 'Overall_Health_Score'],
    r_dir = '/home/mminbay/summer_research/summer23_aylab/Rscripts/',
)

sa.scheirer_ray_hare(
    '/home/mminbay/jonathan/overall_pair_snp_gene_data.csv',
    'pair',
    'Sex',
)

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

