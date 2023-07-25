import os

f = open('/home/mminbay/summer_research/summer23_aylab/data/snps/all_intervals.txt', 'r')

line = f.readline()

prev_chrom = -1
i = 1
while line:
    chrom = int(line.split(',')[0])
    if chrom not in [4, 5, 9, 11, 12, 17]:
        line = f.readline()
        continue
    if chrom != prev_chrom:
        prev_chrom = chrom
        i = 1
    else:
        i += 1
    n = open('/home/mminbay/summer_research/summer23_aylab/data/snps/gene_by_gene/c{}_i{}.txt'.format(chrom, i), 'w')
    n.write(line.strip())
    n.close()
    line = f.readline()

f.close()
        
    