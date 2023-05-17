install.packages('MFSIS')

library(MFSIS)
library(readr)
tcga_data <- read_csv("./raw_data/tcga/tcga_data.csv")

X <- tcga_data[-c(2:5)]
Y <- unlist(tcga_data[2])

n = dim(X)[1]
A <- MVSIS(X, Y, round(n/log(n)))

A

