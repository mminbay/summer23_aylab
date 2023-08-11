import pandas as pd
import os

CHROMS_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/FAULTY_dominant_model/'
OUT_DIR = '/home/mminbay/summer_research/summer23_aylab/data/imputed_data/dominant_model_final/'

TO_KEEP = {'ID_1', 'Sex', 'PHQ9_binary'}

def main():
    files = [file for file in os.listdir(CHROMS_DIR) if '.csv' in file]
    for i in range(len(files)):
        df_drop = pd.read_csv(os.path.join(CHROMS_DIR, files[i]), index_col = 0)
        for j in range(len(files)):
            if j == i:
                continue
            df_check = pd.read_csv(os.path.join(CHROMS_DIR, files[j]), index_col = 0, nrows = 1)
            set_drop = set(df_drop.columns.tolist())
            set_check = set(df_check.columns.tolist())
            if len(set_drop) < len(set_check):
                continue
            cols_todrop = set_drop.intersection(set_check)
            cols_todrop = cols_todrop.difference(TO_KEEP)
            if len(cols_todrop) == 0:
                continue
            df_drop.drop(columns = cols_todrop, inplace = True)
        df_drop.to_csv(os.path.join(OUT_DIR, files[i]))
            
if __name__ == '__main__':
    main()