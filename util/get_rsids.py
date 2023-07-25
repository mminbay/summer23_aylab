import numpy as np
import pandas as pd

df = pd.read_csv('/home/mminbay/summer_research/summer23_aylab/data/snps/allDepressionSNPs.csv')

chroms = df['chrom'].apply(lambda x: x.replace('chr', ''))
rsids = df['name']

sum = 0

f = open('/home/mminbay/summer_research/summer23_aylab/data/snps/all_depression_snps.txt', 'w')
for i in range(len(chroms)):
    if str(chroms[i]) == '1':
        sum += 1
    f.write(chroms[i] + ', ' + rsids[i] + '\n')

print(sum)
f.close()