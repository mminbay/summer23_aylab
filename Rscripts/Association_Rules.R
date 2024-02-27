packages <- c("arules", "wordcloud", "arulesViz", "jpeg", "igraph", "dplyr", "Rgraphviz", "tidygraph", "ggraph", "stringr", "BiocManager", "remotes")

# Function to check and install packages
install_missing_packages <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if (length(new_packages) > 0) {
    install.packages(new_packages, dependencies = TRUE, repos = "http://cran.us.r-project.org")
    cat("Installed the following packages:", paste(new_packages, collapse = ", "), "\n")
  } else {
    cat("All required packages are already installed.\n")
  }
}

# Call the function to install missing packages
install_missing_packages(packages)
library(arules)
library(wordcloud)
library(arulesViz)
library(jpeg)
# library(tiff)
library(igraph)
library(dplyr)
# library(Rgraphviz)
library(tidygraph)
library(ggraph)
library(stringr)
library(BiocManager)
library(remotes)

BiocManager::install("Rgraphviz")
remotes::install_github('adamlilith/legendary', dependencies=TRUE)
library(legendary)

# get data and parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

path <- args[1]
fname <- paste(path,'/AprioriData.csv',sep="")
data <- read.csv(fname, check.names = FALSE)
data <- data[,!names(data) %in% c("X")]
data <- data.frame(lapply(data, as.logical))

sup <- as.numeric(args[2])
con <- as.numeric(args[3])
max <- as.numeric(args[4])
min <- as.numeric(args[5])
varOfInterest <- args[6]
minLift <- as.numeric(args[7])
out_file <- args[8]
center <- args[9]

if (varOfInterest != 'none'){
  rules <- apriori(data, parameter = list(supp=sup, conf=con, maxlen=max, minlen = min, target ="rules"),
                   appearance = list(default="lhs",rhs=varOfInterest))
} else {
  rules <- apriori(data, parameter = list(supp=sup, conf=con, maxlen=max, minlen = min, lift = 3, target ="rules"))
}

rules <- sort(rules, decreasing = TRUE, na.last = NA, by = "lift")
rules <- subset(rules, subset = lift > minLift) # only get rules above lift cutoff

rules.df = DATAFRAME(rules)
write.csv(rules.df, out_file, row.names = FALSE)

lhs = labels(lhs(rules))
rules.df = DATAFRAME(rules)
lift.vals = rules.df$lift
count.vals = rules.df$count
lhs = gsub('[{}]', '', lhs)
list.of.rules = strsplit(lhs, ",")
lhs = paste(lhs, collapse = ",")
lhs = unlist(strsplit(lhs, ","))
LHS = data.frame(lhs = lhs)

# ------------------------------- SNP PLOT ---------------------------- # 

frequency_table = count(LHS, lhs)
feature_names = sort(frequency_table$lhs)
count_matrix = matrix(data = 0, nrow = length(feature_names),
                      ncol = length(feature_names))

colnames(count_matrix) = feature_names
rownames(count_matrix) = feature_names

lift_matrix <- count_matrix

lift_total = data.frame(variable = feature_names,
                        total_lift = rep(0, length(feature_names)),
                        count =  rep(0, length(feature_names)))

for (i in 1:length(list.of.rules)){
  temp.vec = list.of.rules[[i]]
  if (length(temp.vec) != 1){
    for (feature in temp.vec){
      lift_total$total_lift[which(lift_total$variable == feature)] = lift_total$total_lift[which(lift_total$variable == feature)] + lift.vals[i]
        lift_total$count[which(lift_total$variable == feature)] = lift_total$count[which(lift_total$variable == feature)] + 1
    }
    for (j in 1:(length(temp.vec)-1)){
      for (k in (j+1):length(temp.vec)){
        count_matrix[temp.vec[j], temp.vec[k]] = 
          count_matrix[temp.vec[j], temp.vec[k]] + 1
        lift_matrix[temp.vec[j], temp.vec[k]] = 
          lift_matrix[temp.vec[j], temp.vec[k]] + lift.vals[i]
      }
    }
  }
}
# calculate average lift for individual nodes
lift_total <- lift_total %>% mutate(avg_lift = total_lift/count)

# add the lower triangle to the upper triangle for consistency
count_matrix <- t(count_matrix) + count_matrix
lift_matrix <- t(lift_matrix) + lift_matrix
for (col in colnames(count_matrix)) {
    count_matrix[col, col] <- count_matrix[col, col] / 2
    lift_matrix[col, col] <- lift_matrix[col, col] / 2
}

# set lower triangle to zero
count_matrix[lower.tri(count_matrix, diag = FALSE)] <- 0
lift_matrix[lower.tri(lift_matrix, diag = FALSE)] <- 0

# change edge cutoff to your liking
edge_cutoff <- sum(count_matrix) / 100
count_matrix[count_matrix < edge_cutoff] <- 0
lift_matrix[count_matrix < edge_cutoff] <- 0
#initialize graph and layout
graph = igraph::graph_from_adjacency_matrix(count_matrix, mode = "undirected",
                                    diag = F, weighted = T)
coords = layout_(graph, as_star(center = center))

# change node labels so that there are no "." characters
current_node_names <- V(graph)$name
new_node_names <- sub(".", " (", current_node_names, fixed = TRUE)
new_node_names <- sub(".", ")", new_node_names, fixed = TRUE)
V(graph)$label <- new_node_names
print(V(graph)$label)

# normalize edge weights - you can change the max and min however you like
edge_range_min = 2
edge_range_max = 15

min_weight <- min(E(graph)$weight)
max_weight <- max(E(graph)$weight)

edge_width <- edge_range_min + (E(graph)$weight - min_weight) / (max_weight - min_weight) * (edge_range_max - edge_range_min)

# calculate colors
color_palette <- colorRamp(c("gray", "blue")) # you can pick your colors for the gradient

node_lifts = lift_total$avg_lift
edge_lifts <- numeric(0) # calculate edge colors by iterating over the adjacency matrices
i <- 1
for(row in rownames(lift_matrix)) {
    for(col in colnames(lift_matrix)) {
        if (lift_matrix[row, col] > 0) {
            edge_avg_lift <- lift_matrix[row, col] / count_matrix[row, col]
            edge_lifts[[i]] <- edge_avg_lift
            i <- i + 1
        }
    }
}

all_lifts <- c(node_lifts, edge_lifts) # combine node averages and edge averages to maintain same color scale for both
all_lifts_min <- min(all_lifts)
all_lifts_max <- max(all_lifts)
all_lifts_norm <- (all_lifts - all_lifts_min)/(all_lifts_max - all_lifts_min) # normalize average lifts

colors_rgb <- color_palette(all_lifts_norm) # get color representations of normalized average lift on gradient
colors_hex <- c()
for (i in 1:nrow(colors_rgb)){
    colors_hex[i] <- rgb(colors_rgb[i, 1], colors_rgb[i, 2], colors_rgb[i, 3], maxColorValue = 255)
}
node_colors <- colors_hex[1:length(node_lifts)] 
edge_colors <- colors_hex[length(node_lifts) + 1:length(colors)]

# start graphics and plot
jpeg(filename = sub("\\.csv$", "_snp_graph.jpeg", out_file), width = 10, height = 8,
         units = "in", res =300)

plot(graph, layout = na.omit(coords),
     edge.width = edge_width,
     edge.color = edge_colors,
     normalize = F,
     edge.arrow.mode = 0, 
     vertex.color = node_colors, 
     vertex.size =proportions(lift_total$count)*40+5, # you can change this formula if you want
     vertex.label.cex = 0.8, # you should probably make this larger
     vertex.label.font = 2, 
     vertex.label.color= "black",
     vertex.label.dist = case_when(
         coords[, 2] == 0 & coords[, 1] == 0 ~ 3,
         TRUE ~ 1.5 + abs(coords[, 1]) * 2.5
     ), 
     vertex.label.degree = case_when(
         coords[, 2] == 0 & coords[, 1] == 0 ~ pi/4,
         coords[, 1] >= 0 ~ -atan(coords[, 2] / coords[, 1]),
         TRUE ~ -atan(coords[, 2] / coords[, 1]) + pi
     ), 
     margin = c(0, 0, 0, 0.75)
)

legendGrad(
    'right',
    vert = TRUE,
    col = c("gray", "blue"), # make sure the legend colors are the same as your gradient colors
    labels = c(
        round(seq(from = all_lifts_min, to = all_lifts_max, length.out = 5), digits = 2)
    ),
    titleCex = 1.5,
    labCex = 1.5,
    height = 0.6,
    width = 0.05,
    title = "Average Lift",
    inset = 0.05,
    boxBorder = NULL
)
  
dev.off()

# ------------------------------- GENE PLOT ---------------------------- # 

individual_names <- unique(lhs)
coalesced_names <- gsub("_[0-9]+$", "", individual_names)
coalesced_counts <- table(coalesced_names)
for (i in 1:length(coalesced_names)) {
  repr <- coalesced_names[i]
  count <- coalesced_counts[repr]
  coalesced_names[i] <- ifelse(count > 1, paste(repr, " (", count, ")", sep = ""), repr)
}
feature_to_repr <- setNames(coalesced_names, individual_names)
coalesced_names <- sort(unique(coalesced_names))

coalesced_matrix <- matrix(data = 0, nrow = length(coalesced_names),
                      ncol = length(coalesced_names))

colnames(coalesced_matrix) = coalesced_names
rownames(coalesced_matrix) = coalesced_names

coalesced_lift_matrix <- coalesced_matrix

coalesced_lift_total <- data.frame(feature = coalesced_names,
                        total_lift = rep(0, length(coalesced_names)),
                        count = rep(0, length(coalesced_names)))

for (i in 1:length(list.of.rules)){
  current_rule <- list.of.rules[[i]]
  if (length(current_rule) <= 1) {
    next
  }
  for (item in current_rule){
    item_repr <- feature_to_repr[item]
    row <- which(coalesced_lift_total$feature == item_repr)
    coalesced_lift_total$total_lift[row] = coalesced_lift_total$total_lift[row] + lift.vals[i]
    coalesced_lift_total$count[row] = coalesced_lift_total$count[row] + 1
    
  }
  for (j in 1:(length(current_rule) - 1)){
    j_repr <- feature_to_repr[current_rule[j]]
    for (k in (j+1):length(current_rule)){
        k_repr <- feature_to_repr[current_rule[k]]
        coalesced_matrix[j_repr, k_repr] <- coalesced_matrix[j_repr, k_repr] + 1
        coalesced_lift_matrix[j_repr, k_repr] <- coalesced_lift_matrix[j_repr, k_repr] + lift.vals[i]
    }
  }
}

coalesced_lift_total <- coalesced_lift_total %>% mutate(avg_lift = total_lift/count)
coalesced_matrix <- t(coalesced_matrix) + coalesced_matrix
coalesced_lift_matrix <- t(coalesced_lift_matrix) + coalesced_lift_matrix
for (col in colnames(coalesced_matrix)) {
    coalesced_matrix[col, col] <- coalesced_matrix[col, col] / 2
    coalesced_lift_matrix[col, col] <- coalesced_lift_matrix[col, col] / 2
}
coalesced_matrix[lower.tri(coalesced_matrix, diag = FALSE)] <- 0
coalesced_lift_matrix[lower.tri(coalesced_lift_matrix, diag = FALSE)] <- 0

# change edge_cutoff to eliminate more or less weak genes
edge_cutoff <- sum(coalesced_matrix) / 100

coalesced_matrix[coalesced_matrix < edge_cutoff] <- 0
coalesced_lift_matrix[coalesced_matrix < edge_cutoff] <- 0

# create graph and layout
coalesced_graph <- igraph::graph_from_adjacency_matrix(coalesced_matrix, mode = "undirected",
                                            diag = TRUE, weighted = T)
coords <- layout_(coalesced_graph, as_star(center = center))

# change node names
current_node_names <- V(coalesced_graph)$name
new_node_names <- sub(".", " (", current_node_names, fixed = TRUE)
new_node_names <- sub(".", ")", new_node_names, fixed = TRUE)
V(coalesced_graph)$label <- new_node_names

# vertex and edge colors
node_lifts <- coalesced_lift_total$avg_lift
edge_lifts <- numeric(0)
i <- 1
for(row in rownames(coalesced_lift_matrix)) {
    for(col in colnames(coalesced_lift_matrix)) {
        if (coalesced_lift_matrix[row, col] > 0) {
            edge_avg_lift <- coalesced_lift_matrix[row, col] / coalesced_matrix[row, col]
            edge_lifts[[i]] <- edge_avg_lift
            i <- i + 1
        }
    }
}

all_lifts <- round(c(node_lifts, edge_lifts), digits = 2)
all_lifts_min <- min(all_lifts)
all_lifts_max <- max(all_lifts)
# manipulating min and max values to ensure same scale - you normally shouldn't do this
# all_lifts_min <- 2.48
# all_lifts_max <- 6.89
all_lifts_norm <- (all_lifts - all_lifts_min)/(all_lifts_max - all_lifts_min)

colors_rgb <- color_palette(all_lifts_norm)
colors_hex <- c()

for (i in 1:nrow(colors_rgb)){
    colors_hex[i] <- rgb(colors_rgb[i, 1], colors_rgb[i, 2], colors_rgb[i, 3], maxColorValue = 255)
}

node_colors <- colors_hex[1:length(node_lifts)]
edge_colors <- colors_hex[length(node_lifts) + 1:length(colors_rgb)]

# normalize edge weights
edge_range_min = 2
edge_range_max = 15

min_weight <- min(E(coalesced_graph)$weight)
max_weight <- max(E(coalesced_graph)$weight)

edge_width <- edge_range_min + (E(coalesced_graph)$weight - min_weight) / (max_weight - min_weight) * (edge_range_max - edge_range_min)

# calculate edge loop angles
edgeloopAngles <- numeric(0)
b <- 1
M <- nrow(coalesced_matrix)
m <- -1

for(row in rownames(coalesced_matrix)) {
    for(col in colnames(coalesced_matrix)) {
        if (row == col) {
            m <- m+1
        }
        if (coalesced_matrix[row,col] > 0) {
            edgeloopAngles[[b]] <- 0

            if (row == col) {
                edgeloopAngles[[b]] <- ifelse(m >= M/4 && m <= 3*M/4, pi, 0)
            }
            b <- b + 1
        }
    }
}

# start graphics and plot
jpeg(filename = sub("\\.csv$", "_gene_graph.jpeg", out_file), width = 10, height = 8,
     units = "in", res =300)

plot(coalesced_graph, layout = na.omit(coords),
     edge.width = edge_width,
     edge.loop.angle = edgeloopAngles,
     edge.color = edge_colors,
     loop.radius = 0.5,
     normalize = F,
     edge.arrow.mode = 0, vertex.color = node_colors, vertex.size = proportions(coalesced_lift_total$count)*40+5,
     curved = TRUE,
     vertex.label.cex = 1.6, vertex.label.font = 2, vertex.label.color= "black",
     vertex.label.dist = case_when(
         coords[, 2] == 0 & coords[, 1] == 0 ~ 3,
         TRUE ~ 1.5 + abs(coords[, 1]) * 2.5
     ), 
     vertex.label.degree = case_when(
         coords[, 2] == 0 & coords[, 1] == 0 ~ pi/4,
         coords[, 1] >= 0 ~ -atan(coords[, 2] / coords[, 1]),
         TRUE ~ -atan(coords[, 2] / coords[, 1]) + pi
     ), 
     margin = c(0, 0, 0, 0.75)
)

legendGrad(
    'right',
    vert = TRUE,
    col = c("gray", "blue"),
    labels = c(
        round(seq(from = all_lifts_min, to = all_lifts_max, length.out = 5), digits = 2)
    ),
    titleCex = 1.5,
    labCex = 1.5,
    height = 0.6,
    width = 0.05,   
    inset = 0.05,
    title = "Average Lift",
    boxBorder = NULL
)  
dev.off()



