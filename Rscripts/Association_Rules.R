
if(!require(arules)){install.packages("arules", repos = 'http://cran.us.r-project.org')}
if(!require(arulesViz)){install.packages("arulesViz", repos = 'http://cran.us.r-project.org')}
if(!require(RColorBrewer)){install.packages("RColorBrewer", repos = 'http://cran.us.r-project.org')}
if(!require(DescTools)){install.packages("DescTools", repos = 'http://cran.us.r-project.org')}
if(!require(FSA)){install.packages("FSA", repos = 'http://cran.us.r-project.org')}

library(arules)
library(arulesViz)
library(RColorBrewer)
library(htmlwidgets)

args <- commandArgs(trailingOnly = TRUE)

path <- args[1]
fname <- paste(path,'/AprioriData.csv',sep="")
data <- read.csv(fname)
data <- data[,!names(data) %in% c("X")]
data <- data.frame(lapply(data, as.logical))

sup <- as.numeric(args[2])
con <- as.numeric(args[3])
max <- as.numeric(args[4])
min <- as.numeric(args[5])
varOfInterest <- args[6]
minLift <- as.numeric(args[7])
out_file <- args[8]

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

pdf_path <- sub("\\.csv$", "_matrix_based.pdf", out_file)

pdf(file = pdf_path)
plot(rules, method = "grouped")
dev.off()

rules.df = DATAFRAME(rules)
write.csv(rules.df, out_file, row.names = FALSE)

# network_path <- sub("\\.csv$", "_network.html", out_file)

# saveWidget(plot(rules, method = "graph",  engine = "htmlwidget"), 
#            file = network_path)
