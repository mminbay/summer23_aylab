# uncomment this if you need to install packages
# install.packages("SNPassoc", repos = "http://cran.us.r-project.org") # nolint
# install.packages("dplyr",  repos = "http://cran.us.r-project.org") # nolint

library("SNPassoc")
library("dplyr")


###########################################
#1 pre processing
#########

# reading the data file with all the important SNPs, as well as the covariating clinical factors of importance # nolint

# Read the CSV file, replace path with your file path
snpdata <- read.csv("data/analysis_data.csv")

# Select column names starting with "rs". this selects all the SNP columns that start with rs # nolint

snp_column_names <- names(snpdata)[grepl("^rs", names(snpdata))]
# alternatively, you can provide the column numbers for the snps by replacing the previous line # nolint
# with the following (replace the x and y with the start and end cols of where your snps are) # nolint
# snp_identifiers <- names(data)[x:y] # nolint

# Get column numbers corresponding to SNP names
snp_column_numbers <- which(names(snpdata) %in% snp_column_names)

start <- min(snp_column_numbers)
end <- max(snp_column_numbers)
# Create a range of column numbers in the form [1:10]. make sure they are consecutive # nolint
column_range <- paste0("[", start, ":", end, "]") # nolint

# Print the selected column names and column range. this shows you where the snps are in your file # nolint
print(snp_column_names)
print(column_range)


#######################
#2 Using binarized PHQ9
###############

# change clinical factors and target as you want. these are the columnn names
clinical <- c("Sex", "Age", "Chronotype", "Sleeplessness_Insomnia", "TSDI")
# change target to just "PHQ9" for continuous, or whatever your target column name is
target <- "PHQ9_binary"

snpcols <- colnames(snpdata)[min(snp_column_numbers):max(snp_column_numbers)]

# following snippet changes 0 to AA, 1 to AB
# since we have already binarized our data according to dominant model
# this part is used to make our data suit the input data requirements
# since dominant model compares AA vs (AB and BB) we can choose to represent
# 1 as either AB or BB as both of them are converted to 1 in dominant model
snpdata <- snpdata %>%
  mutate_at(vars(start:end),
            list(~ ifelse(. == 0, "AA", ifelse(. == 1, "AB", .))))


# following makes the setupSNP object which is required in the function we will use # nolint
# it takes clinical factors, target (e.g. PHQ9), and names of cols which have the snps as arguments # nolint
data.snp <- setupSNP(data = snpdata %>%
                       select(all_of(c(snp_column_names, clinical, target))),
                     colSNPs = 1:length(snp_column_names), sep = "")

# the interactionPval function, also change the target here if not PHQ9_binary
result.snp = interactionPval(as.formula(paste("PHQ9_binary~",
                                          paste(clinical, collapse="+"))), 
                         data.snp, model = "do")

#################################
#3 outputting the results to a csv
#################
# change the file path to save the result files there
write.csv(result.snp, file = "data/snp-snp_interaction.csv", row.names = TRUE) # nolint
png("data/snp-snp_interaction.png", width = 8, height = 6, units = "in", res = 300)  # Adjust the resolution as needed # nolint
par(mar = c(4, 4, 2, 1))  # Adjust the margin values as needed
plot(result.snp)
dev.off()
