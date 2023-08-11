import os
import numpy as np
import pandas as pd
import logging
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
        out_folder = '/path/to/out_folder/',
        verbose = False
    ):
        '''
        Arguments:
            genetics_folder (str) -- path to the directory where this instance will look for .bgen files (imputed genetic info)
            genetics_format (str) -- name format for .bgen files, where {} will be replaced by chromosome number. note the data field 
            imputed_ids_path (str) -- path to the .csv file which contains the participant id's in the same order as they appear in the .bgen files
            out_folder (str) -- path tho the directory where this instance will write files to
            verbose (bool) -- if True, will log progress of imputed data fetching
        '''
        self.genetics_folder = genetics_folder
        self.genetics_format = genetics_format
        self.imputed_ids_path = imputed_ids_path
        self.out_folder = out_folder
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(filename= os.path.join(out_folder, 'data_loader.log'), encoding='utf-8', level=logging.DEBUG)

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
            file (str) -- path to the .txt file containing the SNP information. The .txt file should have the format <chrom_number>, <rsid> for every line
    
        Modified instance variables:
            self.chroms (list(int)) -- list of chromosomes 
            self.rsids (list(list(str))) -- nested list of rsids on each chromosome. rsids[i] is the list of SNPs of chroms[i].
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
            file (str) -- path to the .txt file containing the interval information. the file should have the format <chrom_number>, <interval_start>-<interval_end> for every line
    
        Modified instance variables:
            self.chroms (list(int)) -- list of chromosomes 
            intervals (list(list(str))) -- nested list of intervals on each chromosome. intervals[i] is the list of intervals of chroms[i].
        '''
        if self.verbose:
            logging.info('Loading chroms and intervals from {}'.format(file))
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
        NOT SUPPORTED! WILL BE UPDATED
        Uses ukbb_parser to get non-imputed genetic information on stored rsids and chroms
    
        Returns:
            df (DataFrame) -- dataset of non-imputed SNP information, where column 'ID_1' is ukbb participant id
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
        NOT SUPPORTED! WILL BE UPDATED
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
        keep = range(1, 23),
        ignore = [],
        n_threshold = 0,
        freq_threshold = 0.0,
        table_name = 'imputed_{}',
        export = True,
        keep_track_as = 'path',
        use_list = False,
        get_alleles = False
    ):
        '''
        TODO: Fix threshold to work with KEPT SAMPLE ONLY
        Uses bgen.reader to get imputed genetic information on stored rsids and chroms only for the ID_1s on given dataframe.
        Saves results into separate .csv for each chromosome.

        Arguments:
            data (str) -- path to .csv file containing the ID_1's to use. the file should have an ID_1 column
            extra (int) -- number of base pairs subtracted from interval starts and added to interval ends to stretch search intervals
            keep (list(int)) -- chromosomes to keep. chromosomes not present in this list will be ignored
            ignore (list(int)) -- chromosome's to ignore. chromosomes in this list will be ignored.
            n_threshold (int) -- SNPs that appear in less than this many people will be discarded.
            freq_threshold (float) -- SNPs that appear in less than this ratio of the population will be discarded.
            table_name (str) -- name format for dataframe for each chromosome, where {} will be chromosome number. will also be used for .csv export file names
            export (bool) -- if True, output the data of each chromosome on its own as a .csv (keep True)
            keep_track_as ('path', 'table', '') -- how this instance will store each chromosome data in self.tables. pass empty string for no storing. note that 'path' option is useless if file isn't being exported. don't do table if you are doing multiple chromosomes
            use_list (bool) -- if True, only get information on rsids that have been provided. you need to call both load_chroms_and_rsids AND load_chroms_and_intervals, and the chroms, intervals, and lists should obviously have no conflicting information.
                for this to work properly, and both sources should obviously have the same chromosomes.
            get_alleles (bool) -- if True, will output an additional .csv file that contains the allele information for all snps that have been recorded
        '''
        ids_to_keep = set(pd.read_csv(data, usecols = ['ID_1'])['ID_1'].tolist()) # this takes way too long if you don't make it a set
        all_ids = pd.read_csv(self.imputed_ids_path, usecols = ['ID_1'])['ID_1'].tolist()
        
        if self.verbose:
            logging.info('Getting imputed information for {} participants'.format(str(len(ids_to_keep))))

        # create list of kept ids in order 
        ordered_kept_ids = [eid for eid in all_ids if eid in ids_to_keep]
        
        chroms = self.chroms
        intervals = self.intervals

        if use_list:
            rsids = self.rsids

        for i in range(len(chroms)):
            current_chrom = chroms[i]
            current_intervals = intervals[i]

            if current_chrom in ignore or current_chrom not in keep:
                if self.verbose:
                    logging.info('Ignoring chromosome {} due to passed parameters'.format(str(current_chrom)))
                continue

            logging.info('Getting information on chromosome {}'.format(str(current_chrom)))

            rows = []
            indices = []
    
            if get_alleles:
                alleles = []
                minor_alleles = []

            if use_list:
                current_rsids = set(rsids[i]) # again, make this a set to save some time

            bfile = bf(os.path.join(self.genetics_folder, self.genetics_format.format(str(current_chrom))), delay_parsing = True)
            
            z = 0
            current_interval = current_intervals[z]
            interval_start = int(current_interval.split('-')[0]) - extra
            interval_end = int(current_interval.split('-')[1]) + extra
            if verbose:
                logging.info('Looking for SNPs in gene interval: {}'.format(str(current_interval)))

            for var in bfile:
                current_position = var.pos
                logging.info('Found a SNP at {}'.format(str(current_position)))
                no_intervals_left = False
                while current_position > interval_end:
                    if self.verbose:
                        logging.info('SNP ahead of interval, shifting interval...')
                    z += 1
                    if z >= len(current_intervals):
                        no_intervals_left = True
                        break
                    current_interval = current_intervals[z]
                    interval_start = int(current_interval.split('-')[0]) - extra
                    interval_end = int(current_interval.split('-')[1]) + extra
                    if self.verbose:
                        logging.info('Looking for SNPs in gene interval: {}'.format(str(current_interval)))
                if no_intervals_left:
                    if self.verbose:
                        logging.info('Ran out of intervals on {}'.format(current_chrom))
                    break
                if current_position < interval_start:
                    if self.verbose:
                        logging.info('SNP before interval, checking next SNP')
                    continue
                if (current_position >= interval_start) and (current_position <= interval_end):
                    variant = var.rsid
                    if self.verbose:
                        logging.info('{} is located in a search interval. Appending...'.format(variant))
                    if use_list:
                        if variant not in current_rsids:
                            continue
                    indices.append(variant)
                    probabilities = var.probabilities
                    if len(probabilities) != len(all_ids):
                        raise Exception('List of ids should match probabilities')
                    row = []
                    sum_snp = 0
                    sum_all = 0
                    for j in range(len(probabilities)):
                        sum_all += 1
                        if probabilities[j].argmax() > 0:
                            sum_snp += 1
                        if all_ids[j] in ids_to_keep:
                            row.append(probabilities[j].argmax())
                    if sum_snp / sum_all < freq_threshold or sum_snp < n_threshold:
                        continue
                    rows.append(row)

                    if get_alleles:
                        alleles.append(str(var.alleles))
                        minor_alleles.append(var.minor_allele)
            if self.verbose:            
                logging.info('Done looking in chromosome {}'.format(str(current_chrom)))

            df = pd.DataFrame(rows, index = indices, columns = ordered_kept_ids)
            df = df.transpose()
            df['ID_1'] = df.index.astype(int)
            formatted_name = table_name.format(chroms[i])
            if export:
                out_file = os.path.join(self.out_folder, formatted_name + '.csv')
                df.to_csv(out_file)
                if self.verbose:
                    logging.info('Exported chromosome {} as a .csv'.format(str(current_chrom)))
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
        extra = 6000, 
        table_name = 'imputed_{}',
        export = True,
        keep_track_as = 'path'
    ):
        '''
        NOT SUPPORTED! WILL BE UPDATED
        Uses bgen.reader to get imputed genetic information on stored chroms and intervals.

        Arguments:
            extra (int) -- number of base pairs subtracted from interval starts and added to interval ends to stretch search intervals
            table_name (str) -- name format for dataframe for each chromosome, where {} will be chromosome number. will also be used for .csv export file names
            export (bool) -- if True, output the data of each chromosome on its own as a .csv (keep True)
            keep_track_as ('path', 'table', '') -- how this instance will store each chromosome data in self.tables. pass empty string for no storing. note that 'path' option is useless if file isn't being exported.
        '''

        # if someone is refactoring this code please use a better bgen module, half the functions in this one do not even work
        chroms = self.chroms
        intervals = self.intervals

        for i in range(len(chroms)):
            current_chrom = chroms[i]
            current_intervals = intervals[i] # the intervals we need to check for this chromosome

            rows = []
            indices = []
            
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
            table_name (str) -- the key with which this table will be stored in self.tables
            binary_cutoff (int) -- used to calculate the PHQ9_binary column, where total PHQ9 scores greater than this cutoff are considered depressed
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
        
    def create_41270_table(self, table_name, values):
        '''
        Uses ukbb_parser to find participant ID's which have any of given codings on their 41270 field.
        Outputs two .csv files as a result: one with a single column of ID_1's of samples with queried values, and another with the entire set of unfiltered results

        Arguments:
            table_name (str) -- the key with which this table will be stored in self.tables, and the name of the output .csv files
            values (list(str)) -- list of values that will be looked for in the 41270 fields. a participant's outcome will be 
                labeled 1 if they have any of these values, 0 otherwise
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
            table_name (str) -- the key with which this table will be stored in self.tables
            columns (list(tuple)) -- list of tuples of the format ('field name', ukbb_field_number, 'data type')
        '''
        eid, fields, _ = create_dataset(
            columns,
            parse_dataset_covariates_kwargs={"use_genotyping_metadata": False},
        )
        fields["ID_1"] = eid

        to_drop = []
        for column, _, _ in columns:
            to_drop.append(column)
        self.tables[table_name] = fields

    def load_table(self, table_name, table_path, delay_parsing = False):
        '''
        Read the .csv file given at table_path and store it at self.tables

        Arguments:
            table_name (str) -- the key with which this table will be stored in self.tables
            table_path (str) -- path to .csv file
            delay_parsing (bool) -- if True, save the path instead of dataframe object and only open .csv when in use.
        '''
        if delay_parsing:
            self.tables[table_name] = table_path
        else:
            self.tables[table_name] = pd.read_csv(table_path)

    def get_table(self, table_name):
        '''
        Return table that is identified with the key 'table_name' in self.tables

        Arguments:
            table_name (str) -- the key with which the table is stored in self.tables

        Returns:
            table (DataFrame) -- the table with the given table_name
        '''

        return self.tables[table_name]

    def merge_all(self, on_col = 'ID_1'):
        '''
        Merge all stored dataframes on given column, and return it.

        Arguments:
            on_col (str) -- column identifier to merge the tables on. unless you really tampered with the code, this is 'ID_1'

        Returns:
            final_table (DataFrame) -- merged dataset of all tables stored in self.tables
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

    @staticmethod
    def export_four_way(
        data,
        bin_outcome,
        cont_outcome,
        ohe,
        out_name,
        out_folder
    ):
        '''
        TO BE IMPLEMENTED
        Export provided dataset in four formats (binary outcome, continuous outcome) x (one-hot-encoded, non-ohe) as separate .csv files
        '''
        pass

'''
Below is an example usage
'''

# def main():
#     dl = DataLoader(
#         genetics_folder = '/datalake/AyLab/data1/ukb_genetic_data',
#         genetics_format = 'ukb22828_c{}_b0_v3.bgen',
#         imputed_ids_path = '/datalake/AyLab/data1/ukb_genetic_data/ukb22828_c1_b0_v3_s487159.csv',
#         out_folder = '/home/mminbay/summer_research/summer23_aylab/data/'
#     ) # create dataloader

#     print('created dataloader')
    
#     clinical_factors = [
#         ("Sex", 31, "binary"),
#         ("Age", 21022, "continuous"),
#         ("Chronotype", 1180, "continuous"),
#         ("Sleeplessness/Insomnia", 1200, "continuous"),
#     ] # an example phenotype dataset

#     dl.create_table('clinical_factors', clinical_factors) # make dataset

#     dl.calcPHQ9('PHQ9_scores', binary_cutoff = 10) # make PHQ9 dataset

#     cf_table = dl.get_table('clinical_factors')
#     phq_table = df.get_table('PHQ9_scores')
#     cf_table = cf_table.dropna()
#     phq_table = phq_table.dropna()

#     final = cf_table.merge(phq_table, on = 'ID_1')

#     final.to_csv('/home/mminbay/summer_research/summer23_aylab/data/depression_data.csv')

#     dl.load_chroms_and_intervals('/home/mminbay/summer_research/summer23_aylab/data/snps/gene_by_gene/c15_i1.txt')

#     dl.get_imputed_from_intervals_for_ids(
#         '/home/mminbay/summer_research/summer23_aylab/data/depression_data.csv',
#         extra = 0,
#         keep = [15],
#         table_name = 'c{}_i1',
#         export = True,
#         keep_track_as = 'path',
#         use_list = False,
#         get_alleles = True
#     )

# if __name__ == "__main__":
#     main()





