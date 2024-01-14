#install.packages('hdm')
library(hdm)
library(readr)
source("residual.balance.R")
source("TMLE.R")

num <- 10
results <- c('error_dse_in' = 0, 'error_arbe_in' = 0, 'error_tmle_linear_in' = 0, 'error_tmle_ensemble_in' = 0,
             'error_dse_out' = 0, 'error_arbe_out' = 0, 'error_tmle_linear_out' = 0, 'error_tmle_ensemble_out' = 0, 
             'sd_dse_in' = 0, 'sd_arbe_in' = 0, 'sd_tmle_linear_in' = 0, 'sd_tmle_ensemble_in' = 0,
             'sd_dse_out' = 0, 'sd_arbe_out' = 0, 'sd_tmle_linear_out' = 0, 'sd_tmle_ensemble_out' = 0)

eps.dse.train <- rep(0,10)
eps.dse.test <- rep(0,10)

eps.arbe.train <- rep(0,10)
eps.arbe.test <- rep(0,10)

eps.tmle.linear.train <- rep(0,10)
eps.tmle.linear.test <- rep(0,10)

eps.tmle.ensemble.train <- rep(0,10)
eps.tmle.ensemble.test <- rep(0,10)

# experiment for ACIC
ate.true = c(0.8, -0.8, -0.3429, 0, -1.432, 9.134, -3.159, -0.8486,-0.16058, 1)

for (i in c(1:num)){
  print(i)
  
  data <- read_csv(paste("../raw_data/acic/acic", i, ".csv", sep=""))
  colnames(data)[which(names(data) == 'Y')] <- 'y'
  colnames(data)[which(names(data) == 'A')] <- 'treat'
  
  # split dataset
  set.seed(10)
  sample <- sample(c(TRUE, FALSE),  replace=TRUE, nrow(data), prob=c(0.66,0.34))
  train <- data[sample, ]
  test <- data[!sample, ]
  
  # reformat data
  y.train <- unlist(train['y'])
  y.test <- unlist(test['y'])
  
  treat.train <- unlist(train['treat'])
  treat.test <- unlist(test['treat'])
  
  x.train <- as.matrix(train[, !(colnames(train) %in% c('y', 'treat'))])
  x.test <- as.matrix(test[, !(colnames(train) %in% c('y', 'treat'))])
  
  # Double selection estimator (DSE)
  set.seed(10)
  
  out.mod <- rlasso(y.train~x.train)
  treat.mod <- rlassologit(treat.train ~ x.train)
  
  out.res.train <- y.train - predict(out.mod, x.train)
  out.res.test <- y.test - predict(out.mod, x.test)
  
  treat.res.train <- treat.train - predict(treat.mod, x.train)
  treat.res.test <- treat.test - predict(treat.mod, x.test)
  
  ate.dse.train <- sum(out.res.train * treat.res.train)/sum(treat.res.train^2)
  ate.dse.test <- sum(out.res.test * treat.res.test)/sum(treat.res.test^2)
  
  eps.dse.train[i] <- abs(ate.dse.train - ate.true[i])
  eps.dse.test[i] <- abs(ate.dse.test - ate.true[i])
  
  print(eps.dse.train[i])
  print(eps.dse.test[i])
  
  # Approximately residual balancing estimators (ARBE)
  set.seed(10)
  
  ate.arbe <- residualBalance.ate(x.train, y.train, x.test, y.test, treat.train, treat.test)
  ate.arbe.train <- ate.arbe[1]
  ate.arbe.test <- ate.arbe[2]
  
  eps.arbe.train[i] <- abs(ate.arbe.train - ate.true[i])
  eps.arbe.test[i] <- abs(ate.arbe.test - ate.true[i])
  
  print(eps.arbe.train[i])
  print(eps.arbe.test[i])
  
  # Targeted maximum likelihood estimators - linear (TMLE)
  set.seed(10)
  
  sl_libs <- c('SL.glmnet', 'SL.glm')
  ate.linear.tmle <- TMLE(y.train, y.test, train[2:dim(train)[2]], test[2:dim(test)[2]], sl_libs)
  ate.tmle.linear.train <- ate.linear.tmle[1]
  ate.tmle.linear.test <- ate.linear.tmle[2]
  
  eps.tmle.linear.train[i] <- abs(ate.tmle.linear.train - ate.true[i])
  eps.tmle.linear.test[i] <- abs(ate.tmle.linear.test - ate.true[i])
  
  print(eps.tmle.linear.train[i])
  print(eps.tmle.linear.test[i])
  
  # Targeted maximum likelihood estimators - ensemble (TMLE)
  set.seed(10)

  sl_libs <- c('SL.glmnet', 'SL.xgboost')
  ate.ensemble.tmle <- TMLE(y.train, y.test, train[2:dim(train)[2]], test[2:dim(test)[2]], sl_libs)
  ate.tmle.ensemble.train <- ate.ensemble.tmle[1]
  ate.tmle.ensemble.test <- ate.ensemble.tmle[2]

  eps.tmle.ensemble.train[i] <- abs(ate.tmle.ensemble.train - ate.true[i])
  eps.tmle.ensemble.test[i] <- abs(ate.tmle.ensemble.test - ate.true[i])

  print(eps.tmle.ensemble.train[i])
  print(eps.tmle.ensemble.test[i])
}

results['error_dse_in'] = mean(eps.dse.train)
results['error_arbe_in'] = mean(eps.arbe.train)
results['error_tmle_linear_in'] = mean(eps.tmle.linear.train)
results['error_tmle_ensemble_in'] = mean(eps.tmle.ensemble.train)

results['error_dse_out'] = mean(eps.dse.test)
results['error_arbe_out'] = mean(eps.arbe.test)
results['error_tmle_linear_out'] = mean(eps.tmle.linear.test)
results['error_tmle_ensemble_out'] = mean(eps.tmle.ensemble.test)

results['sd_dse_in'] = sqrt(var(eps.dse.train)/num)
results['sd_arbe_in'] = sqrt(var(eps.arbe.train)/num)
results['sd_tmle_linear_in'] = sqrt(var(eps.tmle.linear.train)/num)
results['sd_tmle_ensemble_in'] = sqrt(var(eps.tmle.ensemble.train)/num)

results['sd_dse_out'] = sqrt(var(eps.dse.test)/num)
results['sd_arbe_out'] = sqrt(var(eps.arbe.test)/num)
results['sd_tmle_linear_out'] = sqrt(var(eps.tmle.linear.test)/num)
results['sd_tmle_ensemble_out'] = sqrt(var(eps.tmle.ensemble.test)/num)

results

saveRDS(results, file='result/semipara_acic.rds')
