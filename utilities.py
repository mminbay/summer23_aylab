#from ukbb_parser import get_chrom_raw_marker_data
import pandas as pd
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from scipy.stats import multinomial
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold

'''
class utilities:

    def __init__(self):
        pass
    
    def findInRange(self,chromNumber,low,high,ext,exonRanges):
        bim, fam, G = get_chrom_raw_marker_data(str(chromNumber))
        positions=bim['pos'].tolist()
        rss=bim['snp'].tolist()
        rel_rss=[]
        dists=[]
        in_exons=[]
        return_pos=[]
        for i in range(len(positions)):
            pos=positions[i]
            if pos>low-ext and pos<high+ext:
                rel_rss.append(rss[i].strip())
                if pos>low and pos<high:
                    dists.append(0)
                else:
                    dists.append(min(abs(low-pos),abs(high-pos)))
                inExon=0
                for r in exonRanges:
                    if pos>r[0] and pos<r[1]:
                        inExon=1
                        break
                in_exons.append(inExon)
                return_pos.append(pos)

        return [rel_rss, dists, in_exons,return_pos]

    def writeSegRss(self,file,seg,rss,dists,in_exons,poss):
        file.write(seg+':\n'+str(len(rss))+' total, '+str(sum(in_exons))+' in exons\n\n')
        file.write('\t\trsid\t\t\tposition\t\tdist\t\tin_exon\n\n')
        for i in range(len(rss)):
            file.write('\t\t'+rss[i]+'\t\t'+str(poss[i])+'\t\t'+str(dists[i])+'\t\t'+str(in_exons[i])+'\n')
        file.write('\n\n')

    def getExonRanges(self,seg):
        df=pd.read_csv('ExonEdges/'+seg+'.bed',sep='\t',comment='\t',header=None)
        exonRanges=[]
        for i in range(len(df)):
            exonRanges.append((df.iloc[i,1],df.iloc[i,2]))
        return exonRanges

    def writePotentialFile(self):
        geneSegRanges={'CRY1':(12,107385143,107487327),'CRY2':(11,45868669,45904799),'PER1':(17,8043790,8055722),'PER2':(2,239152685,239197251),'PER3':(1,7844351,7905237),'CLOCK3111':(4,56294070,56413076),'ZBTB20':(3,114033347,114343792)}
        genesSegs=['PER3','PER2','ZBTB20','CLOCK3111','CRY2','CRY1','PER1']
        ext=5000

        file=open('potentialRss.txt','w')
        for seg in genesSegs:
            inputs=geneSegRanges[seg]
            exonRanges=self.getExonRanges(seg)
            [rss, dists, in_exons,return_pos]=self.findInRange(inputs[0],inputs[1],inputs[2],ext,exonRanges)
            self.writeSegRss(file,seg,rss,dists,in_exons,return_pos)
        file.close()


#not currently used
def snpIncluded(self,search,chrom):
    bim, fam, G = get_chrom_raw_marker_data(str(chrom))
    for i in range(len(bim)):
        if bim.iat[i,1]==search:
            return True
    return False
'''


def HEOMmod(ucat,unoncat,vcat,vnoncat):

    return math.sqrt(sum(abs(unoncat-vnoncat)**2)+sum((ucat-vcat).astype(bool)*1))


def HEOMmodWeights(ucat,unoncat,vcat,vnoncat,weights):

    return math.sqrt(sum(abs(unoncat-vnoncat)**2)+sum((ucat-vcat).astype(bool)*1*weights))


def HEOM(u,v,cat_masks):

    ucat=u[cat_masks]
    vcat=v[cat_masks]
    unoncat=u[~cat_masks]
    vnoncat=v[~cat_masks]

    return math.sqrt(sum(abs(unoncat-vnoncat)**2)+sum((ucat-vcat).astype(bool)*1))

def remove_underrepresented(input,to_drop=None,levels=None):
    if (input.__class__.__module__,input.__class__.__name__)==('data_processor','data_processor'):
        cols=input.df.columns
        levels=input.levels
        data=input.df
    elif levels is not None:
        cols=input.columns
        data=input
    else:
        print('"levels" parameter must have a non-None value if input is not data_processor object')
        return None

    if to_drop is None:
        to_drop=[]
        for i in range(len(cols)):
            if levels[i]==2:
                one_count=np.sum(data[cols[i]])
                zero_count=len(data)-one_count
                thresh=round(0.001*len(data))
                if one_count<thresh or zero_count<thresh:
                    to_drop.append(cols[i])

    if (input.__class__.__module__,input.__class__.__name__)==('data_processor','data_processor'):
        for title in to_drop:
            input.remove(title)
    else:
        data.drop(labels=to_drop,axis=1,inplace=True)
    
    return to_drop


def replaceNum(num,alleles,single_SNP=True):
    combos={0:'00',1:'01',2:'10',3:'11',4:'02',5:'20',6:'22',7:'12',8:'21'}

    if single_SNP:
        if num=='0':
            return alleles[0]+alleles[0]
        elif num=='1':
            return alleles[0]+alleles[1]
        else:
            return alleles[1]+alleles[1]
    else:
        nums=combos[int(num)]
        num1=nums[0]
        num2=nums[1]
        return_value=''
        if num1=='0':
            return_value += alleles[0][0]+alleles[0][0]
        elif num1=='1':
            return_value += alleles[0][0]+alleles[0][1]
        else:
            return_value += alleles[0][1]+alleles[0][1]

        return_value += 'and'

        if num2=='0':
            return_value += alleles[1][0]+alleles[1][0]
        elif num2=='1':
            return_value += alleles[1][0]+alleles[1][1]
        else:
            return_value += alleles[1][1]+alleles[1][1]

        return return_value

def relabel(OGnode):
    mapping={'rs228697':['PER3A','C','G'],'rs10462020':['PER3C','T','G'],'rs10462023':['PER2','G','A'],'rs1801260':['CLOCK3111','A','G'],'rs17031614':['PER3B','G','A'],'rs139459337':['ZBTB20','C','T'],'rs10838524':['CRY2','A','G'],'rs2287161':['CRY1','C','G']}
    
    node = OGnode.replace(' ', '_')
    node = node.replace('.', '_')
    is_rsid=False
    two_rsids=False
    for key in mapping:
        if key in node:
            if is_rsid:
                two_rsids=True
            is_rsid=True
    if is_rsid:
        new_node=''
        if two_rsids:
            rsid=node[:node.find('_')]
            new_node+=mapping[rsid][0]+'_'
            
            if node.count('_')==3: #is OHC
                new_node+=replaceNum(node[node.find('_')+1],mapping[rsid][1:3])+'_'
                rsid=node[node.find('_')+3:node.rfind('_')]
                new_node+=mapping[rsid][0]+'_'+replaceNum(node[node.rfind('_')+1],mapping[rsid][1:3])
            elif node.count('_')==2:
                rsid1=rsid
                rsid=node[node.find('_')+1:node.rfind('_')]
                new_node+=mapping[rsid][0]+'_'+replaceNum(node[-1],[mapping[rsid1][1:3],mapping[rsid][1:3]],single_SNP=False)               
            else:
                rsid=node[node.find('_')+1:]
                new_node+=mapping[rsid][0]
        else:
            if node.count('_')==1:
                rsid=node[:node.find('_')]
                new_node+=mapping[rsid][0]+'_'+replaceNum(node[node.find('_')+1],mapping[rsid][1:3])
            else:
                new_node+=mapping[node][0]

    else:
        new_node=node

    return new_node


#with william's correction
def G_test_fit(mat):
    n=0
    Sum=0
    key=0
    probs=[]
    counts=[]
    for f in mat:
        if(f[1]==0 or f[0]==0):
            key=1
        if key==0:
            Sum+=f[0]*np.log(f[0]/f[1])
        
        counts.append(f[0])
        n+=f[0]
    
    if key==0:
        G=2*Sum
        q=1+(len(mat)+1)/(6*n)
        return [1,G/q]
    else:
        if n==0:
            return [np.nan,1]
        for f in mat:
            probs.append(f[1]/n)
        if np.isnan(multinomial.pmf(counts,n=n,p=probs)):
            return [np.nan,1]
        return [np.nan,multinomial.pmf(counts,n=n,p=probs)]

def make_frequency_table(df,levels,types):
    df=df.copy(deep=True)
    features=np.array(df.columns)
    cont_idxs=np.where(np.array(types)!='c')[0]
    continuous=list(features[cont_idxs])

    df.drop(labels=continuous,axis=1,inplace=True)
    levels=np.array(levels)[np.setdiff1d(np.array([i for i in range(len(levels))]),cont_idxs)]

    output=pd.DataFrame([[0 for i in range(len(df.columns))] for j in range(max(levels))],index=['Category '+str(i) for i in range(max(levels))], columns=df.columns.copy())
    
    columns=df.columns.copy()
    #print(levels)
    #print(columns)
    for i in range(len(levels)):
        for j in range(levels[i]):
            output.iat[j,np.where(columns==columns[i])[0][0]]=len(np.where(df[columns[i]]==j)[0])

    return output

def ElasticNet(data,target):
    X, y = data.loc[:,np.array([feature for feature in data.columns if feature!=target])], data[target]
    cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=462)
    model=ElasticNetCV(n_jobs=-1)
    model.fit(X,y)
    coefs=model.coef_
    med=np.median(np.abs(coefs))
    print(med)
    bools=np.array(np.abs(coefs)>1.5*med)
    reduced_cols=np.array(data.columns[data.columns!=target])
    print(reduced_cols)
    #print(np.append(reduced_cols[bools],target))
    data=data.loc[:,np.append(reduced_cols[bools],target)]
    return data

