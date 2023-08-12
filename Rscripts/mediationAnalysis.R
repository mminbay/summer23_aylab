if (!requireNamespace("mediation", quietly = TRUE)) {
  install.packages("mediation", repos = 'http://cran.us.r-project.org')
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = 'http://cran.us.r-project.org')
}
# if (!requireNamespace("mmabig", quietly = TRUE)) {
#   install.packages("mmabig", repos = 'http://cran.us.r-project.org')
# }

library(mediation)
library(dplyr)
# library(mmabig)

args <- commandArgs(trailingOnly = TRUE)
path <- args[1]
treatment <- args[2]
mediator <- args[3]
mediator_type <- args[4]
outcome_var <- args[5]
outcome_type <- args[6]
sims <- strtoi(args[7])
covariates <- args[8]
print(covariates)

df <- read.csv(path ,row.names = 1, header = TRUE)

# mediatior model is always assumed to be binomial
if (mediator_type == "bin") {
    if (covariates == "DNE") {
        print("No covariates!")
        fit.mediator <- glm(as.formula(paste(mediator, "~", treatment)), data = df, family = binomial("probit"))
    } else {
        print("Using covariates!")
        fit.mediator <- glm(as.formula(paste(mediator, "~", treatment, "+", covariates)), data = df, family = binomial("probit"))
    }
} else if (mediator_type == "cont") {
    if (covariates == "DNE") {
        print("No covariates!")
        fit.mediator <- lm(as.formula(paste(mediator, "~", treatment)), data = df)
    } else {
        print("Using covariates!")
        fit.mediator <- lm(as.formula(paste(mediator, "~", treatment, "+", covariates)), data = df)
    }
} else {
    stop("Invalid mediator_type specified. Use 'bin' or 'cont'.")
}
if (outcome_type == "bin") {
    if (covariates == "DNE") {
        print("No covariates!")
        fit.dv <- glm(as.formula(paste(outcome_var, "~", mediator, "+", treatment)),
                data = df, family = binomial("probit"))
    } else {
        print("Using covariates!")
        fit.dv <- glm(as.formula(paste(outcome_var, "~", mediator, "+", treatment, "+", covariates)),
                data = df, family = binomial("probit"))
    }
} else if (outcome_type == "cont") {
    if (covariates == "DNE") {
        print("No covariates!")
        fit.dv <- lm(as.formula(paste(outcome_var, "~", mediator, "+", treatment)),
               data = df)
    } else {
        print("Using covariates!")
        fit.dv <- lm(as.formula(paste(outcome_var, "~", mediator, "+", treatment, "+", covariates)),
               data = df)
    }
} else {
    stop("Invalid outcome_type specified. Use 'bin' or 'cont'.")
}
med.out <- mediate(fit.mediator, fit.dv, treat = treatment, mediator = mediator, robustSE = TRUE, sims = sims)

output_file <- paste(treatment, mediator, outcome_var, "mediation.txt", sep = "_")
output_dir <- dirname(path)
output_path <- file.path(output_dir, output_file)
sink(output_path)
summary(med.out)
sink()
