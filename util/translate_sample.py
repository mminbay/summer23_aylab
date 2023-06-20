import sys
import pandas as pd

if len(sys.argv)<2:
	raise Exception('User must provide file name')

file=open(sys.argv[1],'r')
lines=file.readlines()
file.close()
rows=[]

for line in lines[2:]:
	rows.append(line.strip().split(' '))
df=pd.DataFrame(rows, index=None, columns=lines[0].split(' '))
df.to_csv(sys.argv[1][:sys.argv[1].find('.sample')]+'.csv')	
