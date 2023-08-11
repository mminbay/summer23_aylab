import pandas as pd

def separate_columns(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, index_col = 0)

    # Iterate over each column in the dataframe
    for column in df.columns:
        # Create a new dataframe with only the current column
        column_df = pd.DataFrame(df[column])

        # Define the output filename for the current column
        output_file = f"{column}.csv"

        # Write the column dataframe to a new CSV file
        column_df.to_csv(output_file)

        print(f"Output file '{output_file}' created.")

# Provide the path to your input CSV file
csv_file_path = '/home/mminbay/summer_research/summer23_aylab/data/dom_overall/feat_select/c1_dom_overall/name_bootstraps.csv'

# Call the function to separate columns
separate_columns(csv_file_path)
