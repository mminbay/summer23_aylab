import pandas as pd

def find_sig_snp_interactions(csv_file, output_file=None):
    """
    Find significant positions in a symmetric matrix CSV file.

    Given a CSV file containing a symmetric matrix, this function identifies all positions
    (row-name, column-name) in the upper triangular part (excluding the diagonal) where the
    values are less than 0.05. This is made to work well with the csv output of snpassoc since 
    upper triangular of that output file has epistatic snp-snp interactions needed for
    network analysis.

    Parameters:
        csv_file (str): The file path of the CSV file containing the symmetric matrix.
        output_file (str, optional): The file path where the DataFrame will be saved as CSV.

    Returns:
        pandas DataFrame: A two-column DataFrame, where each row has two SNPs with an edge.
        Structure is fit to be used in network analysis.

    Example:
        csv_file_path = 'path/to/your/csv_file.csv'
        result = find_sig_snp_interactions(csv_file_path)
        print(result)

        # Optionally, save the DataFrame to a CSV file
        output_csv_path = 'path/to/output.csv'
        find_sig_snp_interactions(csv_file_path, output_file=output_csv_path)
    """
    # Rest of the function implementation remains the same
    df = pd.read_csv(csv_file, index_col=0)
    significant_positions = set()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            value = df.iloc[i, j]
            if value < 0.05:
                significant_positions.add((df.index[i], df.columns[j]))

    result_df = pd.DataFrame(list(significant_positions), columns=['SNP1', 'SNP2'])

    if output_file:
        result_df.to_csv(output_file, index=False)

    return result_df


