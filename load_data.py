import os
import numpy as np
import pandas as pd
from ukbb_parser import create_dataset, get_chrom_raw_marker_data
from bgen.reader import BgenFile as bf

cwd = os.getcwd()
OUT_FOLDER = os.path.join(cwd, 'data')

# Helper methods to manage dataframes
def combine_df(df1, df2):
    '''
    Inner merge two given dataframes based on 'ID_1' column (participant id)
    '''
    final = pd.merge(df1, df2, on="ID_1", how="inner")
    return final

def drop(data, column):
    '''
    Drop entries with empty or negative values in given columns on dataframe
    '''
    col = np.where(data.columns == column)[0][0]
    to_drop = []
    for i in range(data.shape[0]):
        if np.isnan(data.iloc[i, col]) or data.iloc[i, col] < 0:
            to_drop.append(i)
    data.drop(data.index[to_drop], inplace=True)

# Actual data processing
def loadSnpAndChroms(file=os.path.join(cwd, "summer_research/summer23_aylab/data/snps", "allSNPs.txt")):
    '''
    Reads given .txt file and returns a list of chromosomes and SNP rsids
    
    Arguments:
    file -- Path to the .txt file containing the SNP information. The .txt file should have the format <chrom_number>, <rsid> for every line

    Returns:
    chroms -- list of chromosomes 
    rsids -- nested list of rsids on each chromosome. rsids[i] is the list of SNPs on the chroms[i] chromosome.
    '''
    chroms, rsids = [], []
    prev_chrom, idx = -1, -1
    f = open(file, "r")
    line = f.readline()
    while line and len(line) > 0:
        info = line.split(", ")
        chrom = int(info[0])
        snp = info[1].strip()
        if prev_chrom >= 0 and prev_chrom == chrom:
            rsids[idx].append(snp)
        else:
            idx += 1
            chroms.append(chrom)
            rsids.append([snp])
            prev_chrom = chrom
        line = f.readline()
    f.close()
    return chroms, rsids

def getGeneticInformation(chroms, rsids):
    '''
    Uses ukbb_parser to get non-imputed genetic information on given SNPs

    Arguments:
    chroms -- list of chromosomes (as returned by loadSnpsAndChroms())
    rsids -- nested list of rsids on each chromosome (as returned by loadSnpsAndChroms())

    Returns:
    df -- a dataframe containing non-imputed SNP information, where column 'ID_1' is ukbb participant id
    '''
    df_rows = []
    index = []
    for i in range(len(chroms)):
        bim, fam, G = get_chrom_raw_marker_data(str(chroms[i]))
        indexes = []
        for j in range(len(bim)):
            if bim.iat[j, 1] in rsids[i]:
                indexes.append(j)
                index.append(bim.iat[j, 1])
        for j in indexes:
            df_rows.append(G[j, :].compute())
    iids = fam["iid"].tolist()
    df = pd.DataFrame(df_rows, index=index, columns=iids)
    df = df.transpose()
    df["ID_1"] = df.index.astype(int)
    return df

def getImputedGeneticInformation(chroms, rsids):
    '''
    Uses bgen.reader to get imputed genetic information on given SNPs. Save results into separate .csv for each chromosome

    Arguments:
    chroms -- list of chromosomes (as returned by loadSnpsAndChroms())
    rsids -- nested list of rsids on each chromosome (as returned by loadSnpsAndChroms())

    '''
    path = os.path.join(cwd, "Data", "ukb22828_c1_b0_v3_s487166.csv")
    df_tmp = pd.read_csv(path, index_col=0)
    for i in range(len(chroms)):
        rows = []
        index = []
        # bgen file should be placed here
        bgenPath = os.path.join(
            cwd,
            "Data/genetics/EGAD00010001226/001/",
            "ukb22828_c{}_b0_v3.bgen".format(chroms[i]),
        )
        bfile = bf(bgenPath)
        map = {rsid: index for index, rsid in enumerate(bfile.rsids())}
        print("chrom: " + str(chroms[i]))
        for j in range(len(rsids[i])):
            rsid = rsids[i][j]
            if rsid in map.keys():
                # print('Found rsid: '+rsid)
                index.append(rsid)
                idx = map[rsid]
                probabilities = bfile[idx].probabilities
                rows.append(probabilities.argmax(axis=1))
            else:
                print("Not found rsid: " + rsid)
        df = pd.DataFrame(rows, index=index, columns=df_tmp["ID_1"])
        df = df.transpose()
        df["ID_1"] = df.index.astype(int)
        outFile = os.path.join(cwd, "Data", "imputed_{}.csv".format(chroms[i]))
        df.to_csv(outFile)

def calcPHQ9(binary_cutoff=10):
    '''
    Uses ukbb_parser to get PHQ9 questionnaire answers of participants.

    Arguments:
    binary_cutoff -- used to calculate the PHQ9_binary column, where total PHQ9 scores greater than this cutoff are considered depressed

    Returns:
    df -- a dataframe containing participants' PHQ9 info, where ID_1 is the participant id, PHQ9 is sum of PHQ9 score, and PHQ9_binary is binarized PHQ9 score according to cutoff
    '''
    test = [
        ("Recent changes in speed/amount of moving or speaking", 20518, "continuous"),
        ("Recent feelings of depression", 20510, "continuous"),
        ("Recent feelings of inadequacy", 20507, "continuous"),
        ("Recent feelings of tiredness or low energy", 20519, "continuous"),
        ("Recent lack of interest or pleasure in doing things", 20514, "continuous"),
        ("Recent poor appetite or overeating", 20511, "continuous"),
        ("Recent thoughts of suicide or self-harm", 20513, "continuous"),
        ("Recent trouble concentrating on things", 20508, "continuous"),
        (
            "Trouble falling or staying asleep, or sleeping too much",
            20517,
            "continuous",
        ),
    ]
    eid, fields, _ = create_dataset(
        test, parse_dataset_covariates_kwargs={"use_genotyping_metadata": False}
    )
    rows_binary = []
    rows_score = []
    for i in range(len(fields)):
        row = fields.iloc[i, :].tolist()
        s = 0
        for j in row:
            if np.isnan(j):
                s = -1
                break
            if j >= 0:
                s += j - 1
        if s == -1:
            rows_binary.append(np.nan)
            rows_score.append(np.nan)
        elif s < binary_cutoff:
            rows_binary.append(0)
            rows_score.append(s)
        else:
            rows_binary.append(1)
            rows_score.append(s)

    fields.drop(
        labels=fields.columns[[i for i in range(len(test))]],
        axis=1,
        inplace=True,
    )
    fields["PHQ9"] = rows_score
    fields["PHQ9_binary"] = rows_binary
    fields["ID_1"] = eid
    return fields

def getClinicalFactor(clinical_factors):
    '''
    Uses ukbb_parser to get participants' clinical factor information

    Arguments:
    clinical_factors -- list of tuples of clinical factors ('field name', ukbb_field_number, 'data type')

    Returns:
    fields -- a dataframe containing participants' clinical information, where ID_1 is participant id
    '''
    eid, fields, _ = create_dataset(
        clinical_factors,
        parse_dataset_covariates_kwargs={"use_genotyping_metadata": False},
    )
    fields["ID_1"] = eid
    for column, _, _ in clinical_factors:
        drop(fields, column)
    return fields

# Your code goes here
def main():
    # right now I am just testing
    chroms, rsids = loadSnpAndChroms()
    # clinical factors
    clinical_factors = [
        ("Sex", 31, "binary"),
        ("Age", 21022, "continuous"),
        ("Chronotype", 1180, "continuous"),
        ("Sleeplessness/Insomnia", 1200, "continuous"),
    ]

    # load data from ukbb
    clinical_factor_df = getClinicalFactor(clinical_factors)
    phq9_df = calcPHQ9()
    non_imputed_data = getGeneticInformation(chroms, rsids)
    data = combine_df(clinical_factor_df, non_imputed_data)
    data = combine_df(data, phq9_df)
    data.to_csv(os.path.join(OUT_FOLDER, 'test.csv'))

if __name__ == '__main__':
    main()




