import os
import numpy as np
import pandas as pd
from ukbb_parser import create_dataset, get_chrom_raw_marker_data, create_ICD10_dataset
from bgen.reader import BgenFile as bf
import re

'''
This class is meant to be used as a way to compile your genetic/phenotype/etc. data into a single .csv 
for easier use in your project. Check an example usage at the end of this file.
'''

class DataLoader():
    # TODO: implement init
    def __init__(
        self,
        genetics_folder = '/path/to/genetics/',
        genetics_format = 'ukb22828_c{}_b0_v3.bgen',
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
        self.tables = {}

    def __reset_tables(self):
        self.tables = {}
        
    # Helper methods to manage dataframes
    def __combine_df(self, df1, df2):
        '''
        Inner merge two given dataframes based on 'ID_1' column (participant id)
        '''
        final = pd.merge(df1, df2, on="ID_1", how="inner")
        return final

    def __drop(self, data, columns):
        '''
        Drop entries with empty or negative values in given columns on dataframe
        '''
        for column in columns:
            to_drop = data[pd.isna(data[column]) | data[column] < 0].index
            data.drop(to_drop, inplace = True)
    
    # Actual data processing
    def load_chroms_and_rsids(self, file):
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

    def load_chroms_and_intervals(self, file):
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
            info = line.split(",")
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

    # TODO: refactor to work with self.tables
    def get_nonimputed_from_rsids(self):
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

    # TODO: refactor to work with self.tables
    def get_imputed_from_rsids(self):
        '''
        Uses bgen.reader to get imputed genetic information on stored rsids and chroms. Save results into separate .csv for each chromosome
        '''
        chroms = self.chroms
        rsids = self.rsids
        
        df_tmp = pd.read_csv(self.imputed_ids_path, index_col=0)
        for i in range(len(chroms)):
            current_chrom = chroms[i]
            rows = []
            index = []
            # bgen file should be placed here
            bgenPath = os.path.join(self.genetics_folder, self.genetics_format.format(str(current_chrom)))
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
            outFile = os.path.join(self.out_folder,  'imputed_{}.csv'.format(str(current_chrom)))
            df.to_csv(outFile)

    def get_imputed_from_intervals_for_ids(
        self, 
        data,
        extra = 6000,
        table_name = 'imputed_{}',
        export = True,
        keep_track_as = 'path',
        use_list = False,
        get_alleles = False
    ):
        '''
        Uses bgen.reader to get imputed genetic information on stored rsids and chroms only for the ID_1s on given dataframe.
        Saves results into separate .csv for each chromosome.

        Arguments:
            data -- path to .csv file containing the ID_1's to use
            extra -- number of base pairs subtracted from interval starts and added to interval ends to stretch search intervals
            table_name -- name format for dataframe for each chromosome, where {} will be chromosome number. will also be used for .csv export file names
            export -- whether each chromosome dataframe should be exported on their own as well
            keep_track_as ('path', 'table', '') -- how this instance will store each chromosome data in self.tables. pass empty string for no storing.
                note that 'path' option is useless if file isn't being exported. don't do table if you are doing multiple chromosomes
            use_list -- if True, only get information on rsids that have been provided. you need to call both load_chroms_and_rsids AND load_chroms_and_intervals
                for this to work properly, and both sources should obviously have the same chromosomes.
            get_alleles -- if True, will output an additional .csv file that contains the allele information for all snps that have been recorded
        '''
        ids_to_keep = pd.read_csv(data, usecols = ['ID_1'])['ID_1'].tolist()
        all_ids = pd.read_csv(self.imputed_ids_path, usecols = ['ID_1'])['ID_1'].tolist()

        # create list of kept ids in order 
        ordered_kept_ids = [eid for eid in all_ids if eid in ids_to_keep]
        
        chroms = self.chroms
        intervals = self.intervals

        if use_list:
            rsids = self.rsids

        rows = []
        indices = []

        if get_alleles:
            alleles = []
            minor_alleles = []

        for i in range(len(chroms)):
            current_chrom = chroms[i]
            current_intervals = intervals[i]

            if use_list:
                current_rsids = rsids[i]

            bfile = bf(os.path.join(self.genetics_folder, self.genetics_format.format(str(current_chrom))), delay_parsing = True)

            z = 0
            current_interval = current_intervals[z]
            interval_start = int(current_interval.split('-')[0]) - extra
            interval_end = int(current_interval.split('-')[1]) + extra

            for var in bfile:
                current_position = var.pos
                no_intervals_left = False
                while current_position > interval_end:
                    z += 1
                    if z >= len(current_intervals):
                        no_intervals_left = True
                        break
                    current_interval = current_intervals[z]
                    interval_start = int(current_interval.split('-')[0]) - extra
                    interval_end = int(current_interval.split('-')[1]) + extra
                if no_intervals_left:
                    break
                if current_position < interval_start:
                    continue
                if (current_position >= interval_start) and (current_position <= interval_end):
                    variant = var.rsid
                    if use_list:
                        if variant not in current_rsids:
                            continue
                    indices.append(variant)
                    probabilities = var.probabilities
                    if len(probabilities) != len(all_ids):
                        raise Exception('List of ids should match probabilities')
                    row = []
                    for j in range(len(probabilities)):
                        if all_ids[j] in ordered_kept_ids:
                            row.append(probabilities[j].argmax())
                    rows.append(row)

                    if get_alleles:
                        alleles.append(str(var.alleles))
                        minor_alleles.append(var.minor_allele)

            df = pd.DataFrame(rows, index = indices, columns = ordered_kept_ids)
            df = df.transpose()
            formatted_name = table_name.format(chroms[i])
            if export:
                out_file = os.path.join(self.out_folder, formatted_name + '.csv')
                df.to_csv(out_file)
                if get_alleles:
                    df_allele = pd.DataFrame()
                    df_allele['SNP'] = indices
                    df_allele['alleles'] = alleles
                    df_allele['minor_allele'] = minor_alleles
                    out_file_allele = os.path.join(self.out_folder, formatted_name + '_alleles.csv')
                    df_allele.to_csv(out_file_allele)
                if keep_track_as == 'path':
                    self.tables[formatted_name] = out_file
            if keep_track_as == 'table':
                self.tables[formatted_name] = df

    def getImputedGeneticInformationFromIntervals(
        self, 
        extra = 5000, 
        table_name = 'imputed_{}',
        export = True,
        keep_track_as = 'path'
    ):
        '''
        Uses bgen.reader to get imputed genetic information on stored chroms and intervals.

        Arguments:
        extra -- number of base pairs subtracted from interval starts and added to interval ends to stretch search intervals
        table_name -- name format for dataframe for each chromosome, where {} will be chromosome number. will also be used for .csv export file names
        export -- whether each chromosome dataframe should be exported on their own as well
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
            current_intervals = intervals[i] # the intervals we need to check for this chromosome
            
            print('Checking chromosome {} with delay parsing...'.format(str(current_chrom)))
            bfile = bf(os.path.join(self.genetics_folder, self.genetics_format.format(str(current_chrom))), delay_parsing = True)
            for var in bfile:
                position = var.pos
                for j in range(len(current_intervals)):
                    current_interval = current_intervals[j]
                    interval_start = int(current_interval.split('-')[0]) - extra
                    interval_end = int(current_interval.split('-')[1]) + extra
                    if (position >= interval_start) and (position <= interval_end):
                        variant = var.rsid
                        indices.append(variant)
                        probabilities = var.probabilities
                        rows.append(probabilities.argmax(axis=1))
                        break
            df_tmp = pd.read_csv(self.imputed_ids_path, index_col=0)
            df = pd.DataFrame(rows, index=indices, columns=df_tmp["ID_1"])
            df = df.transpose()
            formatted_name = table_name.format(chroms[i])
            if export:
                out_file = os.path.join(self.out_folder, formatted_name + '.csv')
                df.to_csv(out_file)
                if keep_track_as == 'path':
                    self.tables[formatted_name] = out_file
            if keep_track_as == 'table':
                self.tables[formatted_name] = df

        # the following approach doesn't work with delay_parsing = True
        # for i in range(len(chroms)):
        #     current_chrom = chroms[i]
        #     # REMOVE THIS
        #     if current_chrom == 1:
        #         continue
        #     print('Checking chromosome ' + str(current_chrom))
        #     bfile = bf(os.path.join(self.genetics_folder, self.genetics_format.format(str(current_chrom))), delay_parsing = True)
            
        #     positions = bfile.positions()
        #     rsids = bfile.rsids()
            
        #     current_intervals = intervals[i] # the intervals we need to check for this chromosome
        #     dumb implementation:
        #     for k in range(len(positions)):
        #         current_position = positions[k]
        #         for j in range(len(current_intervals)):
        #             current_interval = current_intervals[j]
        #             interval_start = int(current_interval.split('-')[0]) - extra
        #             interval_end = int(current_interval.split('-')[1]) + extra
        #             if (current_position >= interval_start) and (current_position <= interval_end):
        #                 variant = rsids[k]
        #                 indices.append(variant)
        #                 probabilities = bfile[k].probabilities
        #                 rows.append(probabilities.argmax(axis=1))
        #                 break
                    
            
        #     smart implementation: revise later
        #     z = 0 # to keep track of current interval
        #     current_interval = current_intervals[z]
        #     interval_start = int(current_interval.split('-')[0]) - extra
        #     interval_end = int(current_interval.split('-')[1]) + extra
            
        #     for k in range(len(positions)):
        #         current_position = positions[k]
        #         # if this position is later than current interval, try next intervals
        #         while current_position > interval_end:
        #             z += 1
        #             # if no intervals left to check on this chromosome, stop checking
        #             if z >= len(current_intervals):
        #                 break
        #             current_interval = current_intervals[z]
        #             interval_start = int(current_interval.split('-')[0]) - extra
        #             interval_end = int(current_interval.split('-')[1]) + extra
        #         # if this position is earlier than current interval, check the next position
        #         if current_position < interval_start:
        #             continue
        #         # if position isn't earlier or later, it must be within this interval
        #         if (current_position >= interval_start) and (current_position <= interval_end):
        #             variant = rsids[k]
        #             indices.append(variant)
        #             probabilities = bfile[k].probabilities
        #             rows.append(probabilities.argmax(axis=1))
            
        #     df_tmp = pd.read_csv(self.imputed_ids_path, index_col=0)
        #     df = pd.DataFrame(rows, index=indices, columns=df_tmp["ID_1"])
        #     df = df.transpose()
        #     formatted_name = table_name.format(chroms[i])
        #     if export:
        #         out_file = os.path.join(self.out_folder, formatted_name + '.csv')
        #         df.to_csv(out_file)
        #         if keep_track_as == 'path':
        #             self.tables[formatted_name] = out_file
        #     if keep_track_as == 'table':
        #         self.tables[formatted_name] = df

    def dominant_model(self, data):
        '''
        Relabel given dataframe's SNP info according to dominant model. Assumes all and only SNP columns start with 'rs'

        Arguments:
        data -- dataframe to be relabeled. changes are made in place
        '''
        snp_pattern = re.compile('rs.*')
        snp_cols = [col for col in list(data.columns) if snp_pattern.match(col)]
        for snp in snp_cols:
            temp = data[snp]
            data[snp] = [0 if i == 0 else 1 for i in temp]

    def recessive_model(self, data):
        '''
        Relabel given dataframe's SNP info according to recessive model. Assumes all and only SNP columns start with 'rs'

        Arguments:
        data -- dataframe to be relabeled. changes are made in place
        '''
        snp_pattern = re.compile('rs.*')
        snp_cols = [col for col in list(data.columns) if snp_pattern.match(col)]
        for snp in snp_cols:
            temp = data[snp]
            data[snp] = [1 if i == 2 else 0 for i in temp]

    def one_hot_encode(self, data, columns):
        '''
        One-hot encode the given columns on the given dataframe. Returns OHC encoded dataframe

        Arguments:
        data -- dataframe to be modified
        columns -- features to be one-hot encoded

        Returns:
        result -- OHC encoded dataframe
        '''
        return pd.get_dummies(data, columns = columns, drop_first = True)
        

    # TODO: make this general purpose
    def calcPHQ9(self, table_name, binary_cutoff=10):
        '''
        Uses ukbb_parser to create a PHQ9 dataset with given columns. Stores created dataset in self.tables
    
        Arguments:
        table_name -- the key with which this table will be stored in self.tables
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
        self.tables[table_name] = fields
        
    def create_41270_table(self, table_name, values, export = True):
        '''
        Uses ukbb_parser to find participant ID's which have any of given codings on their 41270 field.
        This dataframe will only have 1 column 'ID_1'.

        Arguments:
            table_name -- the key with which this table will be stored in self.tables
            values -- list of values that will be looked for in the 41270 fields. a participant's outcome will be 
                labeled 1 if they have any of these values, 0 otherwise
            export -- whether the resulting dataframes should be exported on their own
        '''
        eid, fields, _ = create_dataset(
            [('41270 fields', 41270, 'raw')],
            parse_dataset_covariates_kwargs={"use_genotyping_metadata": False},
        )
        columns = fields.columns.tolist()
        fields['ID_1'] = eid
        fields.to_csv(os.path.join(self.out_folder, table_name + '.csv'))

        result = fields[(fields.isin(values)).any(axis = 1)]
        result.drop(columns = columns)
        result.to_csv(os.path.join(self.out_folder, table_name + '_depressed_only.csv'))
        self.tables[table_name]
    
    def create_table(self, table_name, columns):
        '''
        Uses ukbb_parser to create a dataset with given columns. Stores created dataset in self.tables
    
        Arguments:
        table_name = the key with which this table will be stored in self.tables
        columns -- list of tuples of the format ('field name', ukbb_field_number, 'data type')
        '''
        eid, fields, _ = create_dataset(
            columns,
            parse_dataset_covariates_kwargs={"use_genotyping_metadata": False},
        )
        fields["ID_1"] = eid

        to_drop = []
        for column, _, _ in columns:
            to_drop.append(column)
        # self.__drop(fields, to_drop)
        self.tables[table_name] = fields

    # TODO: implement properly
    def create_ICD10_table(self, table_name, columns):
        '''
        Uses ukbb_parser to create a dataset with given columns AND ICD10 results. Stores tree and phenotype table in self.tables
        '''
        eid, ICD10_tree, fields, _, _ = create_ICD10_dataset(
            parse_dataset_covariates_kwargs={"use_genotyping_metadata": False}
        )
        ICD10_tree.to_csv('/home/mminbay/summer_research/summer23_aylab/data/test.csv')
        

    def load_table(self, table_name, table_path, delay_parsing = False):
        '''
        Read the .csv file given at table_path and store it at self.tables

        Arguments:
        table_name -- the key with which this table will be stored in self.tables
        table_path -- path to .csv file
        delay_parsing -- if True, save the path instead of dataframe object and only open .csv when in use.
        '''
        if delay_parsing:
            self.tables[table_name] = table_path
        else:
            self.tables[table_name] = pd.read_csv(table_path)

    def get_table(self, table_name):
        '''
        Return table that is identified with the key 'table_name' in self.tables

        Arguments:
        table_name -- the key with which the table is stored in self.tables

        Returns:
        table -- the table with the given table_name
        '''

        return self.tables[table_name]

    def merge_all(self, on_col = 'ID_1'):
        '''
        Merge all stored dataframes on given column, and return it.
        '''
        table_list = list(self.tables.values())
        if len(table_list) == 0:
            raise Exception('No tables to merge and export')

        final_table = table_list[0]
        if type(final_table) == str:
            final_table = pd.read_csv(final_table)
        other_tables = table_list[1:]
        for table in other_tables:
            if type(table) == str:
                table = pd.read_csv(table)
            final_table = final_table.merge(table, on = on_col, how = 'inner')

        self.__drop(final_table, final_table.columns)
        return final_table

    # TODO: refactor to work with only given set of table names
    def export(self, data, out = 'export.csv', drop = [], ohc = [], model = 'd'):
        '''
        Merge stored tables on given column and save as .csv at given path. 

        Arguments:
        data -- dataframe to export
        out -- name of output file
        drop -- list of columns to drop. useful for maintaining a single outcome column per dataframe (check export_all). pass empty string for no drop
        ohc -- list of variables to one hot encode. pass empty list for no encoding
        model ('d', 'r') -- d for dominant model, r for recessive model
        '''
        
        if len(drop) > 0:
            data.drop(columns = drop)
        if len(ohc) > 0:
            data = self.one_hot_encode(final_table, ohc)

        if model.equals('d'):
            self.dominant_model(data)
        elif model.equals('r'):
            self.recessive_model(data)

        data.to_csv(os.path.join(self.out_folder, out))

    def export_all(self, ohc, filename = 'result', binary_outcome = 'PHQ9_binary', continuous_outcome = 'PHQ9', model = 'd'):
        '''
        Merge all stored dataframes. Export 4 dataframes (OHC, non-OHC) * (binary outcome, continuous outcome)
        '''
        final_table = self.merge_all()
        self.export(final_table, out = filename + '_OHC_binary.csv', drop = [continuous_outcome], ohc = ohc, model = 'd')
        self.export(final_table, out = filename + '_OHC_continuous.csv', drop = [binary_outcome], ohc = [], model = 'd')
        self.export(final_table, out = filename + '_noOHC_binary.csv', drop = [continuous_outcome], ohc = [], model = 'd')
        self.export(final_table, out = filename + '_noOHC_continuous.csv', drop = [binary_outcome], ohc = ohc, model = 'd')

'''
Below is an example usage
'''

# OUTDATED! REDO
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

#     dl.make_table('clinical factors', clinical_factors) # make dataset

#     dl.calcPHQ9('PHQ9 scores', binary_cutoff = 10) # make PHQ9 dataset

#     dl.getImputedGeneticInformationFromIntervals(
#         extra = 5000, 
#         export_name = 'haplotype_imputed_{}.csv',
#         keep_track_as = 'table'
#     ) # make imputed genetic info dataset from loaded intervals

#     dl.export(on_col = 'ID_1', out = 'export_test.csv') # export all informartion as a single table

# if __name__ == "__main__":
#     main()





