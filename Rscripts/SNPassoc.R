# uncomment this if you need to install packages
# install.packages("SNPassoc", repos = "http://cran.us.r-project.org") # nolint
# install.packages("dplyr",  repos = "http://cran.us.r-project.org") # nolint

library("SNPassoc")
library("dplyr")

###########################################
#1 pre processing
###########################################

create_snpassoc_paths <- function(csv_path, analysis_type) {
  # Extract the file name, file extension, and parent directory
  file_name <- basename(csv_path)
  parent_dir <- dirname(csv_path)
  extension <- tools::file_ext(csv_path)

  # Determine the file name suffix based on the analysis type
  if (analysis_type == "PHQ9") {
    suffix <- "cont_snpassoc"
  } else if (analysis_type == "PHQ9_binary") {
    suffix <- "bin_snpassoc"
  } else {
    stop("Invalid analysis_type. Please use 'PHQ9' or 'PHQ9_binary'.")
  }

  # Create the new file names
  csv_new_name <- paste0(file_name, "_", suffix, ".csv")
  png_new_name <- paste0(file_name, "_", suffix, ".png")

  # Create the new paths with parent directory
  csv_new_path <- file.path(parent_dir, csv_new_name)
  png_new_path <- file.path(parent_dir, png_new_name)

  return(list(csv_new_path, png_new_path))
}

# reading the data file with all the important SNPs, as well as the covariating clinical factors of importance # nolint

if (length(commandArgs(trailingOnly = TRUE)) != 2) {
  stop("Please provide three command line arguments: Input path, and outcome variable")
}

snpdata_path <- commandArgs(trailingOnly = TRUE)[1]
target <- commandArgs(trailingOnly = TRUE)[2]

output_paths_list <- create_snpassoc_paths(snpdata_path, target)

# Read the CSV file, replace path with your file path
snpdata <- read.csv(snpdata_path)

# Select column names starting with "rs". this selects all the SNP columns that start with rs # nolint

snp_column_names <- names(snpdata)[grepl("^rs", names(snpdata))]
# alternatively, you can provide the column numbers for the snps by replacing the previous line # nolint
# with the following (replace the x and y with the start and end cols of where your snps are) # nolint
# snp_identifiers <- names(data)[x:y] # nolint

# Get column numbers corresponding to SNP names
snp_column_numbers <- which(names(snpdata) %in% snp_column_names)

start <- min(snp_column_numbers)
end <- max(snp_column_numbers)
# manual override for SNP column numbers if not all of them start with rs!:
# snpcol_start <- 
# snpcol_end <- 

# Create a range of column numbers in the form [1:10]. make sure they are consecutive # nolint
column_range <- paste0("[", start, ":", end, "]") # nolint

# Print the selected column names and column range. this shows you where the snps are in your file # nolint
print(snp_column_names)
print(column_range)


#######################
#2 Using binarized PHQ9
###############

# change clinical factors and target as you want. these are the columnn names
clinical <- c("Chronotype_2.0", "Chronotype_3.0", "Chronotype_4.0", "Sleeplessness.Insomnia_2.0", "Sleeplessness.Insomnia_3.0", "Overall_Health_Score_2.0", "Overall_Health_Score_3.0", "Overall_Health_Score_4.0", "TSDI_n")

snpcols <- colnames(snpdata)[min(snp_column_numbers):max(snp_column_numbers)]

# following snippet changes 0 to AA, 1 to AB
# since we have already binarized our data according to dominant model
# this part is used to make our data suit the input data requirements
# since dominant model compares AA vs (AB and BB) we can choose to represent
# 1 as either AB or BB as both of them are converted to 1 in dominant model
# make it so that it is 0 to AA and 1 to BB for recessive model
snpdata <- snpdata %>%
  mutate_at(vars(start:end),
            list(~ ifelse(. == 0, "AA", ifelse(. == 1, "AB", .))))


# following makes the setupSNP object which is required in the function we will use # nolint
# it takes clinical factors, target (e.g. PHQ9), and names of cols which have the snps as arguments # nolint
data.snp <- setupSNP(data = snpdata %>%
                       select(all_of(c(snp_column_names, clinical, target))),
                     colSNPs = 1:length(snp_column_names), sep = "")

# the interactionPval function. Change the target here too (PHQ9 OR PHQ9_binary, or whatever you have)
# if you want to add clinical factors, change the line as: result.snp <- interactionPval(as.formula(paste("PHQ9_binary~", paste(clinical, collapse="+"))), # nolint
# if you want to remove clinical factors, the same line becomes: result.snp <- interactionPval(as.formula(paste("PHQ9_binary~1")), # nolint
result.snp <- interactionPval(as.formula(paste(target, "~1")), # nolint
                             data.snp, model = "do")

#################################
#3 outputting the results to a csv
#################
# change the file path to save the result files there
write.csv(result.snp, file = output_paths_list[[1]], row.names = TRUE) # nolint
png(output_paths_list[[2]], width = 8, height = 6, units = "in", res = 300)  # Adjust the resolution as needed # nolint
par(mar = c(4, 4, 2, 1))  # Adjust the margin values as needed
plot(result.snp)
dev.off()
