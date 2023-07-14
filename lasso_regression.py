import pandas as pd

# Create a dataframe of the csv file
snp_df = pd.read_csv('/datalake/AyLab/depression_study/depression_snp_data/dominant_model/dominant_c15_i1.csv', index_col=0)
clinical_df = pd.read_csv('/datalake/AyLab/depression_study/depression_basket_data/depression_data.csv', usecols = ['ID_1', 'PHQ9'])
merge = snp_df.merge(clinical_df, how = 'inner', on = 'ID_1')

merge = merge.head(100)
merge_copy = merge.head(100)

merge = merge.head(10)
# Assuming you have a DataFrame called 'df' and want to save it to a CSV file named 'output.csv'
merge.to_csv('small_dataset.csv', index=False)

selected_columns = ['PHQ9'] + [col for col in merge.columns if col.startswith('rs')][:10]
merge_subset = merge[selected_columns]

merge_subset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import numpy as np

# Specify the columns to include
included_columns = merge.columns[merge.columns.str.startswith('rs')].tolist()

# Prepare the data
X = merge[included_columns]
y = merge['PHQ9']
feature_names = X.columns  # Store the column names for later use

# Set the minimum number of important features
min_num_features = 50

# Set the number of bootstrap iterations
num_bootstrap = 3

# Create an empty dictionary to store the coefficients for each SNP
coefficients_dict = {}

# Perform bootstrap iterations
num_iterations_with_min_features = 0

while num_iterations_with_min_features < num_bootstrap:
    # Bootstrap resampling
    bootstrap_indices = np.random.choice(len(merge), size=len(merge), replace=True)
    bootstrap_sample = merge.iloc[bootstrap_indices]

    # Prepare the data
    X = bootstrap_sample[included_columns]
    y = bootstrap_sample['PHQ9']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform lasso regression
    lasso = Lasso(alpha=0.001)  # Adjust alpha according to your preference
    lasso.fit(X_train, y_train)

    # Filter out the most important features
    important_features = X.columns[abs(lasso.coef_) > 0]

    # Check if the number of important features is at least the minimum
    if len(important_features) >= min_num_features:
        num_iterations_with_min_features += 1

        # Store the coefficients for each SNP separately
        for feature in important_features:
            if feature in coefficients_dict:
                coefficients_dict[feature].append(abs(lasso.coef_[X.columns == feature][0]))
            else:
                coefficients_dict[feature] = [abs(lasso.coef_[X.columns == feature][0])]

        # Evaluate the model
        train_score = lasso.score(X_train, y_train)
        test_score = lasso.score(X_test, y_test)

        # Print the iteration details
        print(f"\nBootstrap iteration {num_iterations_with_min_features}")
        print("Training R^2 score:", train_score)
        print("Testing R^2 score:", test_score)
        print("Important features:")
        print(important_features)
        print("Number of features:", len(important_features))

# Create a final dataframe for SNPs appearing in at least 2 out of 3 dataframes
final_dataframe = pd.DataFrame(columns=['SNP', 'Average Coefficient'])

# Iterate over the coefficients dictionary
for feature, values in coefficients_dict.items():
    # Check if the SNP appears in at least 2 out of 3 dataframes
    if len(values) >= 2:
        if len(values) == 2:
            # Calculate the average of the two coefficient values
            avg_coefficient = np.mean(values)
        else:
            # Calculate the average of the two highest coefficient values
            two_highest_values = np.sort(values)[-2:]
            avg_coefficient = np.mean(two_highest_values)
            
        # Append the SNP and average coefficient to the final dataframe
        final_dataframe = final_dataframe.append({'SNP': feature, 'Average Coefficient': avg_coefficient},
                                                 ignore_index=True)

# Print the final dataframe
print("\nFinal DataFrame:")
print(final_dataframe)