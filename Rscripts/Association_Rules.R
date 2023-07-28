
#install.packages("arules", repos='http://cran.us.r-project.org')
#install.packages("arulesViz", repos='http://cran.us.r-project.org')
#install.packages("RColorBrewer", repos='http://cran.us.r-project.org')
#install.packages("DescTools")
#install.packages("FSA")


library(arules)
library(arulesViz)
library(RColorBrewer)
library(htmlwidgets)


args <- commandArgs(trailingOnly = TRUE)


path <- args[1]
fname <- paste(path,'AprioriData.csv',sep="")
data <- read.csv(fname)
data <- data[,!names(data) %in% c("X")]
data <- data.frame(lapply(data, as.logical))


sup <- as.numeric(args[2])
con <- as.numeric(args[3])
max <- as.numeric(args[4])
min <- as.numeric(args[5])
minLift <- as.numeric(args[7])
varOfInterest <- args[6]


if (varOfInterest != 'none'){
  rules <- apriori(data, parameter = list(supp=sup, conf=con, maxlen=max, minlen = min, target ="rules"),
                   appearance = list(default="lhs",rhs=varOfInterest))
} else {
  rules <- apriori(data, parameter = list(supp=sup, conf=con, maxlen=max, minlen = min, lift = 3, target ="rules"))
}

rules <- sort(rules, decreasing = TRUE, na.last = NA, by = "lift")
rules <- subset(rules, subset = lift > minLift)

quality(rules)$oddsRatio <-  interestMeasure(rules, measure = "oddsRatio", smoothCounts = 0.5)
quality(rules)$pValue <-  interestMeasure(rules, measure = "fishersExactTest")
quality(rules)$T <- interestMeasure(rules, measure = "table")
rules <- rules[!is.redundant(rules, measure= "oddsRatio")]


pdf(file = paste(path, 'AprioriMatrixBased.pdf', sep = ""))
plot(rules, method = "grouped")
dev.off()

rules.df = DATAFRAME(rules)
write.csv(rules.df,paste(path,"apriori.csv",sep=""), row.names = FALSE)


saveWidget(plot(rules, method = "graph",  engine = "htmlwidget"), 
           file= paste(path,"AprioriNetwork.html",sep=""))
