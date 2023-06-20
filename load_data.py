import os
import numpy as np
import pandas as pd
from ukbb_parser import create_dataset, get_chrom_raw_marker_data
from bgen.reader import BgenFile as bf

'''
This class is meant to be used as a way to compile your genetic/phenotype/etc. data into a single .csv 
for easier use in your project. Check an example usage at the end of this file.
'''

class DataLoader():
    # TODO: implement init
    def __init__(
        self,
        genetics_folder = '/path/to/genetics/',
        genetics_format = 'ukb22438_c{}_b0_v2.bgen',
        imputed_ids_path = '/path/to/imputed/ids.csv',
        out_folder = '/path/to/out_folder/'
    ):
        '''
        Arguments:
        genetics_folder -- where this instace will look for .bgen files (imputed genetic info)
        genetics_format -- name format for .bgen files, where {} will be replaced by chromosome number. note the data field 
        imputed_ids_path -- path to the .csv file which contains the participant id's in the same order as they appear in the .bgen files
        out_folder -- where this instance will write files to
        '''
        self.genetics_folder = genetics_folder
        self.genetics_format = genetics_format
        self.imputed_ids_path = imputed_ids_path
        self.out_folder = out_folder

        # for later reuse with other types of data fields
        self.tables = []

    def __reset_tables(self):
        self.tables = []
        
    # Helper methods to manage dataframes
    def __combine_df(self, df1, df2):
        '''
        Inner merge two given dataframes based on 'ID_1' column (participant id)
        '''
        final = pd.merge(df1, df2, on="ID_1", how="inner")
        return final

    def __drop(self, data, column):
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
    def loadChromsAndRsidsFromList(self, file):
        '''
        Reads given .txt file and stores list of chroms and rsids as instance variables
        
        Arguments:
        file -- Path to the .txt file containing the SNP information. The .txt file should have the format <chrom_number>, <rsid> for every line
    
        Modified instance variables:
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
        self.chroms = chroms
        self.rsids = rsids

    def loadChromsAndRsidsFromInterval(self, file):
        '''
        Reads given .txt file and stores list of chroms and intervals as instance variables
        
        Arguments:
        file -- Path to the .txt file containing the interval information. The .txt file should have the format <chrom_number>, <interval_start>-<interval_end>
        for every line
    
        Modified instance variables:
        chroms -- list of chromosomes 
        intervals -- nested list of intervals on each chromosome. intervals[i] is the list of intervals on the chroms[i] chromosome.
        '''
        chroms, intervals = [], []
        prev_chrom, idx = -1, -1
        f = open(file, "r")
        line = f.readline()
        while line and len(line) > 0:
            info = line.split(", ")
            chrom = int(info[0])
            interval = info[1].strip()
            if prev_chrom >= 0 and prev_chrom == chrom:
                intervals[idx].append(interval)
            else:
                idx += 1
                chroms.append(chrom)
                intervals.append([interval])
                prev_chrom = chrom
            line = f.readline()
        f.close()
        self.chroms = chroms
        self.intervals = intervals
    
    def getGeneticInformationFromList(self):
        '''
        Uses ukbb_parser to get non-imputed genetic information on stored rsids and chroms
    
        Returns:
        df -- a dataframe containing non-imputed SNP information, where column 'ID_1' is ukbb participant id
        '''
        chroms = self.chroms
        rsids = self.rsids
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

    # TODO: re-implement this to work with self.tables
    def getImputedGeneticInformationFromList(self):
        '''
        Uses bgen.reader to get imputed genetic information on stored rsids and chroms. Save results into separate .csv for each chromosome
        '''
        chroms = self.chroms
        rsids = self.rsids

        # TODO: edit this
        path = os.path.join(self.cwd, "Data", "ukb22828_c1_b0_v3_s487166.csv")
        
        df_tmp = pd.read_csv(path, index_col=0)
        for i in range(len(chroms)):
            rows = []
            index = []
            # bgen file should be placed here
            bgenPath = os.path.join(
                self.cwd,
                "Data/genetics/EGAD00010001226/001/",
                "ukb22828_c{}_b0_v3.bgen".format(chroms[i]),
            )
            bfile = bf(bgenPath)
            map = {rsid: index for index, rsid in enumerate(bfile.rsids())}
            for j in range(len(rsids[i])):
                rsid = rsids[i][j]
                if rsid in map.keys():
                    index.append(rsid)
                    idx = map[rsid]
                    probabilities = bfile[idx].probabilities
                    rows.append(probabilities.argmax(axis=1))
                else:
                    print("Not found rsid: " + rsid)
            df = pd.DataFrame(rows, index=index, columns=df_tmp["ID_1"])
            df = df.transpose()
            df["ID_1"] = df.index.astype(int)
            outFile = os.path.join(self.cwd, "Data", "imputed_{}.csv".format(chroms[i]))
            df.to_csv(outFile)

    def getImputedGeneticInformationFromIntervals(
        self, 
        extra = 5000, 
        export_name = 'imputed_{}.csv',
        keep_track_as = 'path'
    ):
        '''
        Uses bgen.reader to get imputed genetic information on stored chroms and intervals.

        Arguments:
        extra -- number of base pairs subtracted from interval starts and added to interval ends to stretch search intervals
        export_name -- name format for export file for each chromosome, where {} will be chromosome number. pass empty string for no export
        keep_track_as ('path', 'table', '') -- how this instance will store each chromosome data in self.tables. pass empty string for no storing.
            note that 'path' option is useless if file isn't being exported.
        '''

        # if someone is refactoring this code please use a better bgen module, half the functions in this one do not even work
        chroms = self.chroms
        intervals = self.intervals

        rows = []
        indices = []

        for i in range(len(chroms)):
            current_chrom = chroms[i]
            bfile = bf(os.path.join(self.genetics_folder, self.genetics_format.format(str(current_chrom))))
            
            positions = bfile.positions()
            rsids = bfile.rsids()
            
            current_intervals = intervals[i] # the intervals we need to check for this chromosome

            z = 0 # to keep track of current interval
            current_interval = current_intervals[z]
            interval_start = int(current_interval.split('-')[0]) - extra
            interval_end = int(current_interval.split('-')[1]) + extra
            
            for k in range(len(positions)):
                current_position = positions[k]
                # if this position is later than current interval, try next intervals
                while current_position > interval_end:
                    z += 1
                    # if no intervals left to check on this chromosome, stop checking
                    if z >= len(current_intervals):
                        break
                    current_interval = current_intervals[z]
                    interval_start = int(current_interval.split('-')[0]) - extra
                    interval_end = int(current_interval.split('-')[1]) + extra
                # if this position is earlier than current interval, check the next position
                if current_position < interval_start:
                    continue
                # if position isn't earlier or later, it must be within this interval
                if current_position >= interval_start and current_position <= interval_end:
                    variant = rsids[k]
                    indices.append(variant)
                    probabilities = bfile[k].probabilities
                    rows.append(probabilities.argmax(axis=1))
            
            df_tmp = pd.read_csv(self.imputed_ids_path, index_col=0)
            df = pd.DataFrame(rows, index=indices, columns=df_tmp["ID_1"])
            df = df.transpose()
            if export_name != '':
                out_file = os.path.join(self.out_folder, export_name.format(chroms[i]))
                df.to_csv(out_file)
                if keep_track_as == 'path':
                    self.tables.append(out_file)
            if keep_track_as == 'table':
                self.tables.append(df)

    # TODO: make this general purpose
    def calcPHQ9(self, binary_cutoff=10):
        '''
        Uses ukbb_parser to create a PHQ9 dataset with given columns. Stores created dataset in self.tables
    
        Arguments:
        binary_cutoff -- used to calculate the PHQ9_binary column, where total PHQ9 scores greater than this cutoff are considered depressed
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
        self.tables.append(fields)
    
    def make_table(self, columns):
        '''
        Uses ukbb_parser to create a dataset with given columns. Stores created dataset in self.tables
    
        Arguments:
        columns -- list of tuples of the format ('field name', ukbb_field_number, 'data type')
        '''
        eid, fields, _ = create_dataset(
            columns,
            parse_dataset_covariates_kwargs={"use_genotyping_metadata": False},
        )
        fields["ID_1"] = eid
        for column, _, _ in columns:
            self.__drop(fields, column)
        self.tables.append(fields)

    def import_table(self, table_path, delay_parsing = False):
        '''
        Read the .csv file given at table_path and store it at self.tables

        Arguments:
        table_path -- path to .csv file
        delay_parsing -- if True, save the path instead of dataframe object and only open .csv when in use.
        '''
        if delay_parsing:
            self.tables.append(table_path)
        else:
            self.tables.append(pd.read_csv(table_path))

    def export(self, on_col = 'ID_1', out = 'export.csv', ohe = []):
        '''
        Merge stored tables on given column and save as .csv at given path. 

        Arguments:
        on_col -- column to merge tables on
        out -- name of output file
        ohe -- variables to one-hot encode
        '''
        if len(self.tables) == 0:
            raise Exception('No tables to merge and export')

        final_table = self.tables[0]
        other_tables = self.tables[1:]
        for table in other_tables:
            if type(table) == str:
                table = pd.read_csv(table)
            final_table = final_table.merge(table, on = on_col, how = 'inner')

        for column in list(final_table.columns):
            self.__drop(final_table, column)

        # TODO: implement one hot encoding!
        final_table.to_csv(os.path.join(self.out_folder, out))

'''
Below is an example usage
'''

# def main():
#     dl = DataLoader(
#         genetics_folder = '/shared/datalake/summer23Ay/data1/ukb_genetic_data',
#         genetics_format = 'ukb22438_c{}_b0_v2.bgen',
#         imputed_ids_path = '/shared/datalake/summer23Ay/data1/ukb_genetic_data/ukb22828_c1_b0_v3_s487166.csv',
#         out_folder = '/home/mminbay/summer_research/summer23_aylab/data/'
#     ) # create dataloader
    
#     dl.loadChromsAndRsidsFromInterval('/home/mminbay/summer_research/summer23_aylab/data/snps/intervals.txt') # load intervals
    
#     clinical_factors = [
#         ("Sex", 31, "binary"),
#         ("Age", 21022, "continuous"),
#         ("Chronotype", 1180, "continuous"),
#         ("Sleeplessness/Insomnia", 1200, "continuous"),
#     ] # an example phenotype dataset

#     dl.make_table(clinical_factors) # make dataset

#     dl.calcPHQ9(binary_cutoff = 10) # make PHQ9 dataset

#     dl.getImputedGeneticInformationFromIntervals(
#         extra = 5000, 
#         export_name = 'haplotype_imputed_{}.csv',
#         keep_track_as = 'table'
#     ) # make imputed genetic info dataset from loaded intervals

#     dl.export(on_col = 'ID_1', out = 'export_test.csv') # export all informartion as a single table

# if __name__ == "__main__":
#     main()





