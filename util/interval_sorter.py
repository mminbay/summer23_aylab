'''
This method takes a .txt file with unsorted intervals, and outputs a new .txt file where they are sorted
'''
import os
from functools import cmp_to_key

def __compare_intervals(line1, line2):
    if int(line1.split(',')[0]) < int(line2.split(',')[0]):
        return -1
    if int(line1.split(',')[0]) > int(line2.split(',')[0]):
        return 1
    if int(line1.split(',')[1].split('-')[0]) < int(line2.split(',')[1].split('-')[0]):
        return -1
    if int(line1.split(',')[1].split('-')[0]) > int(line2.split(',')[1].split('-')[0]):
        return 1
    return -1
    

def sort_intervals(file):
    file = os.path.join(DATA_DIR, file)
    result = []
    f = open(file, "r")
    line = f.readline()
    while line and len(line) > 0:
        print(line.rstrip())
        result.append(line.rstrip())
        line = f.readline()
    f.close()
    result = sorted(result, key=cmp_to_key(__compare_intervals))
    return result

def write_to_file(lines, name):
    out = os.path.join(DATA_DIR, name)
    f = open(out, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()

# your changes go here
DATA_DIR = '/home/mminbay/summer_research/summer23_aylab/data/snps/'

def main():
    write_to_file(sort_intervals('intervals_unsorted.txt'), 'all_intervals.txt')

if __name__ == '__main__':
    main()

    
        