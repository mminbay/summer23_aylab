#install.packages("mediation")
#setwd("C:/Users/colev/OneDrive/Desktop/ukbb_analyzer")
library(mediation)
library("dplyr")
library("mmabig")

args = commandArgs(trailingOnly = TRUE)

df=read.csv(paste("Data/mediationData.csv"),row.names=1,header=TRUE)

treatment=args[1]
mediator=args[2]
outcome_var=args[3]
covariates=args[4]


#covariates=unlist(strsplit(covariates,'\\+'))
#drops=c(treatment,outcome_var)
#d=df[ , !names(df) %in% drops]
#treatment=df[args[1]]
#outcome_var=df[args[3]]
#data.e1.2<-data.org.big(x=d,y=outcome_var,mediator=1:ncol(d),pred=treatment,testtype=1)
#summary(data.e1.2,only=TRUE)

#for (y in colnames(df)){
#    print(y)
#}


fit.mediator=lm(as.formula(paste(mediator,"~",treatment,"+",covariates)),data=df)#family = binomial("probit")
 #fit.mediator=glm(Chronotype_3~rs10838524_1.rs2287161_2,data=df,family = binomial("probit"))
fit.dv=glm(as.formula(paste(outcome_var,"~",mediator,"+",treatment,"+",covariates)),data=df,family = binomial("probit"))
 #fit.dv=glm(GAD7_1~Chronotype_3+rs10838524_1.rs2287161_2,data=df,family = binomial("probit"))
med.out=mediate(fit.mediator,fit.dv,treat=treatment,mediator=mediator,robustSE = TRUE, sims = 100)
sink("mediation.txt")
summary(med.out)
sink()
