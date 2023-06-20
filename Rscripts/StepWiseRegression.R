#install.packages('tidyverse', dependencies = TRUE, repos='http://cran.us.r-project.org')
#install.packages('caret', dependencies = TRUE, repos='http://cran.us.r-project.org')
#install.packages('leaps', dependencies = TRUE, repos='http://cran.us.r-project.org')
#install.packages("My.stepwise", dependencies = TRUE, repos='http://cran.us.r-project.org')

library(MASS)
library(tidyverse)
library(caret)
library(leaps)
library(readr)

#library(My.stepwise)

args = commandArgs(trailingOnly = TRUE)

path = args[1]
#fname = paste(path,'StepWiseRegData.csv',sep="")
fname = paste(path,args[4],sep="")
regType = args[2]
var = args[3]


data = read.csv(fname)
data = data[,!names(data) %in% c("X")]

formula = as.formula(paste(var,'~.',sep=""))
if (regType == 'linear'){
  full.model <- lm(formula, data = data)
  step.model <- stepAIC(full.model, direction = "both", 
                        trace = FALSE)

  
} else if (regType == 'logit') {
  full.model <- glm(formula, data = data, family = "binomial")
  step.model <- stepAIC(full.model, direction = "both", 
                        trace = FALSE)
}


newModel <- model.matrix(step.model)
newVars <- as.list(colnames(newModel)) 
lapply(newVars, write, paste(path,'stepWiseVars.txt',sep=""), append=TRUE, ncolumns=1000)


