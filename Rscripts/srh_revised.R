# note all these tests are being done for dominant group
if(!require(rcompanion)){install.packages("rcompanion", repos = 'http://cran.us.r-project.org')}
if(!require(FSA)){install.packages("FSA", repos = 'http://cran.us.r-project.org')}

library(rcompanion)
library(FSA)

if (length(commandArgs(trailingOnly = TRUE)) != 6) {
    stop("Please provide six command line arguments: Input path, SNP file path, SNP type (single or pair), sex column label, target column label, and output path")
}


input_file <- commandArgs(trailingOnly = TRUE)[1]
snps_file <- commandArgs(trailingOnly = TRUE)[2]
curr_snp <- commandArgs(trailingOnly = TRUE)[3]
sex_column <- commandArgs(trailingOnly = TRUE)[4]
phq9_column <- commandArgs(trailingOnly = TRUE)[5]
output_folder <- commandArgs(trailingOnly = TRUE)[6]

### Assembling the data

my_data <- read.csv(input_file)
snps <- read.csv(snps_file)

# Replace colons in SNP names and column names starting with 'pair'
snps$name <- gsub(":", ".", snps$name)

# Add the if statement to check 'curr_snp'
if (curr_snp == "single") { # here, the mapped file has a name (snp name) and the gene it belongs to #nolint
    selected_snps <- snps[, c("name", "adjusted.pval", "Gene")]
} else { # here the name is the snp pair name, and gene1, gene2 are the two genes that the pair of snps belongs to #nolint
    selected_snps <- snps[, c("name", "adjusted.pval", "Gene1", "Gene2")]
}

# Filter snps based on adjusted p-value <= 0.05
filtered_snps <- selected_snps[selected_snps$adjusted.pval <= 0.05, ]

# Loop through the filtered snps and perform SRH test for each snp_column
for (i in seq_len(nrow(filtered_snps))) {
    snp_name <- filtered_snps[i, "name"]
    if (curr_snp == "single") {
        snp_column <- snp_name
    } else {
        snp_column <- snp_name
    }
    to_analyze <- my_data[, c(snp_column, sex_column, phq9_column)]
    srh_formula <- as.formula(paste(phq9_column, "~", snp_column, "+", sex_column))
    
    srh_results <- scheirerRayHare(formula = srh_formula, data = to_analyze)
    
    # Build the output file name
    if (curr_snp == "single") {
        gene_name <- as.character(filtered_snps[i, "Gene"])
        output_file <- paste(output_folder, sprintf("%s_%s_srh.csv", gene_name, snp_name))
    } else {
        gene1_name <- as.character(filtered_snps[i, "Gene1"])
        gene2_name <- as.character(filtered_snps[i, "Gene2"])
        output_file <- paste(output_folder, sprintf("%s_%s_%s_srh.csv", gene1_name, gene2_name, snp_name))
    }
    
    write.csv(list(Scheirer_Ray_Hare = srh_results), file = output_file)
}