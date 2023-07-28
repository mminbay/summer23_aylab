library(mgm)
library(qgraph)
library(comprehenr)

args=commandArgs(trailingOnly=TRUE)


readfile<-read.csv(file=args[1],row.names=1)
types<-unlist(strsplit(args[2],','))
levels<-as.integer(unlist(strsplit(args[3],',')))



#if(FALSE){
fit_ADS<-mgm(data=as.matrix(readfile),
             type=types,
             level=levels,
             k=2,
             lambdaSel='EBIC',
             lambdaGam=0.25,
             pbar=FALSE)

qgraph(fit_ADS$pairwise$wadj, 
       layout = 'spring', repulsion = 1.3,
       edge.color = fit_ADS$pairwise$edgecolor, 
       nodeNames = colnames(readfile),
       #color = autism_data_large$groups_color, 
       #groups = autism_data_large$groups_list,
       #legend.mode="style2", legend.cex=.45, 
       vsize = 3.5, esize = 15)

write.csv(fit_ADS$pairwise$wadj,file='network_results.csv')
#}