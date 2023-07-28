library(minet)

#args=commandArgs(trailingOnly=TRUE)


#readfile<-read.csv(file=args[1],row.names=1)
readfile<-read.csv(file='data_tmp_16_M.csv',row.names=1)

print(1)
mim=build.mim(data=as.matrix(readfile))

col_names<-colnames(mim)
bound<-nrow(mim)
for(i in 1:bound){
    if(i==bound){
        break
    }
    for(j in i+1:bound){
        if(col_names[i]!="Chronotype" && col_names[i]!="GAD7" && col_names[j]!="Chronotype" && col_names[j]!="GAD7"){
            mim[i,j]<-0
            mim[j,i]<-0
        }
        if(j==bound){
            break
        }
    }
}

print(2)
net<-aracne(mim)
print(3)


write.csv(net,file="R_network_results_16_M.csv")