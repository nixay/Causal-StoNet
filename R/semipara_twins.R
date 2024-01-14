#install.packages('hdm')
library(hdm)
library(readr)
library(caret)
source("residual.balance.R")
source("TMLE.R")

num <- 3 

results <- c('ate_dse' = 0, 'ate_arbe' = 0, 'ate_tmle_linear' = 0, 'ate_tmle_ensemble' = 0,
             'sd_dse' = 0, 'sd_arbe' = 0, 'sd_tmle_linear' = 0, 'sd_tmle_ensemble' = 0)

# experiment for Twins data
dse <- rep(0,3)
arbe <- rep(0,3)
tmle.linear <- rep(0,3)
tmle.ensemble <- rep(0,3)

# read data
data <- read_csv("../raw_data/twins/twins_data.csv")
data <- subset(data, select = -counter)

# shuffle dataset
set.seed(10)
data <- data[sample(1:nrow(data)), ] # shuffle dataset
size <- round(dim(data)[1]/3)

for (cv in 1:num){
  print(cv)
  
  # split dataset
  test.start <- size*(cv-1)+1
  test.end <- min(dim(data)[1], size*cv)
  test.idx <- c(test.start:test.end)
  test <- data[test.idx, ]
  train <- data[-test.idx, ]
  
  # # use SMOTE to create a balanced dataset
  # train <- data.frame(lapply(train, as.factor))
  # train <- smote(y~., train, perc.over = 2, perc.under = 2)
  # train <- train[sample(1:nrow(train)), ]
  # train <- data.frame(lapply(train, function(x) as.numeric(x)-1))
  
  # reformat data
  y.train <- unlist(train$y)
  y.test <- unlist(test$y)
  
  treat.train <- unlist(train$treat)
  treat.test <- unlist(test$treat)
  
  x.train <- as.matrix(train[, !(colnames(train) %in% c('y', 'treat'))])
  x.test <- as.matrix(test[, !(colnames(train) %in% c('y', 'treat'))])
  
  # Double selection estimator (DSE)
  set.seed(10)
  
  out.mod <- rlassologit(y.train~x.train)
  treat.mod <- rlassologit(treat.train ~ x.train)

  out.res.train <- y.train - predict(out.mod, x.train)
  out.res.test <- y.test - predict(out.mod, x.test)

  treat.res.train <- treat.train - predict(treat.mod, x.train)
  treat.res.test <- treat.test - predict(treat.mod, x.test)

  ate.dse.train <- sum(out.res.train * treat.res.train)/sum(treat.res.train^2)
  ate.dse.test <- sum(out.res.test * treat.res.test)/sum(treat.res.test^2)

  print(ate.dse.train)
  print(ate.dse.test)

  dse[cv] <- ate.dse.test

  # Approximately residual balancing estimators (ARBE)
  set.seed(10)
  
  ate.arbe <- residualBalance.ate(x.train, y.train, x.test, y.test, treat.train, treat.test, binary=TRUE, scale.X = FALSE)
  ate.arbe.train <- ate.arbe[1]
  ate.arbe.test <- ate.arbe[2]
  
  print(ate.arbe.train)
  print(ate.arbe.test)
  
  arbe[cv] <- ate.arbe.test
  
  # Targeted maximum likelihood estimators - linear (TMLE)
  set.seed(10)
  
  sl_libs <- c('SL.glmnet', 'SL.glm')
  ate.linear.tmle <- TMLE(y.train, y.test, train[2:dim(train)[2]], test[2:dim(test)[2]], binary=TRUE, sl_libs)
  ate.tmle.linear.train <- ate.linear.tmle[1]
  ate.tmle.linear.test <- ate.linear.tmle[2]

  print(ate.tmle.linear.train)
  print(ate.tmle.linear.test)

  tmle.linear[cv] <- ate.tmle.linear.test
  
  # Targeted maximum likelihood estimators - ensemble (TMLE)
  set.seed(10)

  sl_libs <- c('SL.glmnet', 'SL.xgboost')
  ate.ensemble.tmle <- TMLE(y.train, y.test, train[2:dim(train)[2]], test[2:dim(test)[2]], binary=TRUE, sl_libs)
  ate.tmle.ensemble.train <- ate.ensemble.tmle[1]
  ate.tmle.ensemble.test <- ate.ensemble.tmle[2]

  print(ate.tmle.ensemble.train)
  print(ate.tmle.ensemble.test)

  tmle.ensemble[cv] <- ate.tmle.ensemble.test
}

results['ate_dse'] = mean(dse)
results['ate_arbe'] = mean(arbe)
results['ate_tmle_linear'] = mean(tmle.linear)
results['ate_tmle_ensemble'] = mean(tmle.ensemble)

results['sd_dse'] = sd(dse)/sqrt(num)
results['sd_arbe'] = sd(arbe)/sqrt(num)
results['sd_tmle_linear'] = sd(tmle.linear)/sqrt(num)
results['sd_tmle_ensemble'] = sd(tmle.ensemble)/sqrt(num)

results

saveRDS(results, file='./result/semipara_twins.rds')
