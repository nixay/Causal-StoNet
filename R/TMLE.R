# install.packages('tidyverse')
# install.packages('SuperLearner')
# install.packages('readr')

library(tidyverse)
library(SuperLearner)
library(readr)

TMLE = function(Y.train, Y.test, W_A.train, W_A.test, learner, binary=FALSE){
  sl_libs <- learner
  
  ### Step 1: Estimate Q
  if (binary){
    Q <- SuperLearner(Y = Y.train, # Y is the outcome vector
                      X = W_A.train, # W_A is the matrix of X and A
                      family=binomial(), # continuous outcome
                      SL.library = sl_libs)
  }
  else{
    Q <- SuperLearner(Y = Y.train, # Y is the outcome vector
                      X = W_A.train, # W_A is the matrix of X and A
                      family=gaussian(), # continuous outcome
                      SL.library = sl_libs)
  }
  Q_A.train <- as.vector(predict(Q)$pred) # obtain predictions for everyone using the treatment they actually received
  W_A1.train <- W_A.train %>% mutate(treat = 1)  # data set where everyone received treatment
  Q_1.train <- as.vector(predict(Q, newdata = W_A1.train)$pred) # predict on that everyone-exposed data set
  W_A0.train <- W_A.train %>% mutate(treat = 0) # data set where no one received treatment
  Q_0.train <- as.vector(predict(Q, newdata = W_A0.train)$pred)
  dat_tmle.train <- tibble(Y = Y.train, A = W_A.train$treat, Q_A=Q_A.train, Q_0=Q_0.train, Q_1=Q_1.train)
  
  Q_A.test <- as.vector(predict(Q, W_A.test)$pred)
  W_A1.test <- W_A.test %>% mutate(treat = 1) 
  Q_1.test <- as.vector(predict(Q, newdata = W_A1.test)$pred)
  W_A0.test <- W_A.test %>% mutate(treat = 0) 
  Q_0.test <- as.vector(predict(Q, newdata = W_A0.test)$pred)
  dat_tmle.test <- tibble(Y = Y.test, A = W_A.test$treat, Q_A=Q_A.test, Q_0=Q_0.test, Q_1=Q_1.test)
  
  ### Step 2: Estimate g and compute H(A,W)
  A.train <- W_A.train$treat
  W.train <- W_A.train %>% select(-treat) # matrix of predictors that only contains the confounders
  
  A.test <- W_A.test$treat
  W.test <- W_A.test %>% select(-treat)
  
  g <- SuperLearner(Y = A.train, # outcome is the A (treatment) vector
                    X = W.train, # W is a matrix of predictors
                    family=binomial(), # treatment is a binomial outcome
                    SL.library=sl_libs) # using same candidate learners; could use different learners
  
  g_w.train <- as.vector(predict(g)$pred) # Pr(A=1|W)
  H_1.train <- 1/g_w.train
  H_0.train <- -1/(1-g_w.train) # Pr(A=0|W)
  dat_tmle.train <- # add clever covariate data to dat_tmle
    dat_tmle.train %>%
    bind_cols(
      H_1 = H_1.train,
      H_0 = H_0.train) %>%
    mutate(H_A = case_when(A == 1 ~ H_1, # if A is 1 (treated), assign H_1
                           A == 0 ~ H_0))  # if A is 0 (not treated), assign H_0
  
  g_w.test <- as.vector(predict(g, W.test)$pred)
  H_1.test <- 1/g_w.test
  H_0.test <- -1/(1-g_w.test)
  dat_tmle.test <- 
    dat_tmle.test %>%
    bind_cols(
      H_1 = H_1.test,
      H_0 = H_0.test) %>%
    mutate(H_A = case_when(A == 1 ~ H_1,
                           A == 0 ~ H_0)) 
  
  ### Step 3: Estimate fluctuation parameter
  lm_fit <- lm(Y ~ -1 + offset(Q_A) + H_A, data=dat_tmle.train) # fixed intercept regression
  eps <- coef(lm_fit) # save the only coefficient, called epsilon in TMLE lit
  
  ### Step 4: Update Q's
  Q_A_update.train <- dat_tmle.train$Q_A + eps*dat_tmle.train$H_A # updated expected outcome given treatment actually received
  Q_1_update.train <- dat_tmle.train$Q_1 + eps*dat_tmle.train$H_1 # updated expected outcome for everyone receiving treatment
  Q_0_update.train <- dat_tmle.train$Q_0 + eps*dat_tmle.train$H_0 # updated expected outcome for everyone not receiving treatment
  
  Q_A_update.test <- dat_tmle.test$Q_A + eps*dat_tmle.test$H_A
  Q_1_update.test <- dat_tmle.test$Q_1 + eps*dat_tmle.test$H_1
  Q_0_update.test <- dat_tmle.test$Q_0 + eps*dat_tmle.test$H_0
  
  ### Step 5: Compute ATE
  tmle_ate.train <- mean(Q_1_update.train - Q_0_update.train) # mean diff in updated expected outcome estimates
  
  tmle_ate.test <- mean(Q_1_update.test - Q_0_update.test)
  
  ### Step 6: compute standard error, CIs and pvals
  #infl_fn <- (Y - Q_A_update) * H_A + Q_1_update - Q_0_update - tmle_ate # influence function
  #tmle_se <- sqrt(var(infl_fn)/nrow(dat_obs)) # standard error
  #conf_low <- tmle_ate - 1.96*tmle_se # 95% CI
  #conf_high <- tmle_ate + 1.96*tmle_se
  #pval <- 2 * (1 - pnorm(abs(tmle_ate / tmle_se))) # p-value at alpha .05
  
  return(c(tmle_ate.train, tmle_ate.test))
}
