source("approx.balance.R")

#' Estimate mean outcome at balance.target via residual balancing
#'
#' @param XW_train the input features for the sub-population of interest (training set)
#' @param YW_train the observed responses for the sub-population of interest (training set)
#' @param XW_test the input features for the sub-population of interest (testing set)
#' @param YW_test the observed responses for the sub-population of interest (testing set)
#' @param balance.target.train the desired center of the dataset (training set)
#' @param balance.target.test the desired center of the dataset (test set)
#' @param binary if the outcome variable is binary
#' @param allow.negative.weights whether negative gammas are allowed for balancing
#' @param zeta tuning parameter for selecting approximately balancing weights
#' @param fit.method the method used to fit mu(x) = E[YW | XW = x]
#' @param alpha tuning paramter for glmnet
#' @param optimizer which optimizer to use for approximate balancing
#' @param bound.gamma whether upper bound on gamma should be imposed
#' @param verbose whether the optimizer should print progress information
#'
#' @return Estimate for E[YW | XW = balance.target], along with variance estimate
#'
#' @export residualBalance.mean
residualBalance.mean = function(XW_train, YW_train, XW_test, YW_test,
                                balance.target.train,
                                balance.target.test,
                                binary = FALSE,
                                allow.negative.weights = FALSE,
                                zeta,
                                fit.method = c("elnet", "none"),
                                alpha,
                                optimizer = c("mosek", "pogs", "pogs.dual", "quadprog"),
                                bound.gamma = TRUE,
                                verbose = FALSE) {
  
  fit.method = match.arg(fit.method)
  optimizer = match.arg(optimizer)
  
  gamma.train = approx.balance(XW_train, balance.target.train, zeta = zeta, allow.negative.weights = allow.negative.weights, optimizer = optimizer, bound.gamma=bound.gamma, verbose=verbose)
  gamma.test = approx.balance(XW_test, balance.target.test, zeta = zeta, allow.negative.weights = allow.negative.weights, optimizer = optimizer, bound.gamma=bound.gamma, verbose=verbose)
  
  if (fit.method == "elnet") {
    
    if (binary){
      lasso.fit = glmnet::cv.glmnet(XW_train, YW_train, alpha = alpha, family = 'binomial')
    }
    else{
      lasso.fit = glmnet::cv.glmnet(XW_train, YW_train, alpha = alpha, family = 'gaussian')
    }
    
    mu.lasso.train = predict(lasso.fit, newx = matrix(balance.target.train, 1, length(balance.target.train)))
    mu.lasso.test = predict(lasso.fit, newx = matrix(balance.target.test, 1, length(balance.target.test)))
    
    residuals.train = YW_train - predict(lasso.fit, newx = XW_train)
    mu.residual.train = sum(gamma.train * residuals.train)
    
    residuals.test = YW_test - predict(lasso.fit, newx = XW_test)
    mu.residual.test = sum(gamma.test * residuals.test)
  
    # var.hat = sum(gamma^2 * residuals^2) *
    #   # degrees of freedom correction
    #   length(gamma) / max(1, length(gamma) - sum(coef(lasso.fit) != 0))
    
  } else if (fit.method == "none") {
    
    mu.lasso.train = 0
    mu.lasso.test = 0
    
    mu.residual.train = sum(gamma.train * YW_train)
    mu.residual.test = sum(gamma.test * YW_test)
    
    # var.hat = NA
    
  } else {
    
    stop("Invalid choice of fitting method.")
    
  }
  
  mu.hat.train = mu.lasso.train + mu.residual.train
  mu.hat.test = mu.lasso.test + mu.residual.test
  # c(mu.hat, var.hat)
  c(mu.hat.train, mu.hat.test)
}

# residualBalance.estimate.var = function(XW_train, YW_train, XW_test, YW_test, alpha, estimate.se) {
#   
#   # Don't waste time
#   if (!estimate.se) return(NA)
#   
#   lasso.fit = glmnet::cv.glmnet(XW_train, YW_train, alpha = alpha)
#   residuals = YW_test - predict(lasso.fit, newx = XW_test)
#   mean(residuals^2) / max(1, length(YW_test) - sum(coef(lasso.fit) != 0))
# }


#' Estimate in-sample and out-of-sample ATE via approximate residual balancing
#' 
#' @param XW_train the input features (training set)
#' @param YW_train the observed responses (training set)
#' @param XW_test the input features (test set)
#' @param YW_test the observed responses (test set)
#' @param W_train treatment/control assignment, coded as 0/1 (training set)
#' @param W_test treatment/control assignment, coded as 0/1 (test set)
#' @param binary if the outcome variable is binary
#' @param target.pop which population should the treatment effect be estimated for?
#'            (0, 1): average treatment effect for everyone
#'            0: average treatment effect for controls
#'            1: average treatment effect for treated
#' @param allow.negative.weights whether negative gammas are allowed for balancing
#' @param zeta tuning parameter for selecting approximately balancing weights
#' @param fit.method the method used to fit mu(x, w) = E[Y | X = x, W = w]
#' @param alpha tuning paramter for glmnet
#' @param scale.X whether non-binary features should be noramlized
#' @param optimizer which optimizer to use for approximate balancing
#' @param bound.gamma Whether upper bound on gamma should be imposed. This is
#'             required to guarantee asymptotic normality, but increases computational cost.
#' @param verbose whether the optimizer should print progress information
#'
#' @return ATE estimate, along with (optional) standard error estimate
#'
#' @export residualBalance.ate
residualBalance.ate = function(XW_train, YW_train, XW_test, YW_test, W_train, W_test,
                               binary = FALSE,
                               target.pop=c(0, 1),
                               allow.negative.weights = FALSE,
                               zeta=0.5,
                               fit.method = c("elnet", "none"),
                               alpha=0.9,
                               scale.X = TRUE,
                               optimizer = c("mosek", "pogs", "pogs.dual", "quadprog"),
                               bound.gamma = TRUE,
                               verbose = FALSE) {
  
  fit.method = match.arg(fit.method)
  optimizer = match.arg(optimizer)
  
  # if (estimate.se & fit.method == "none") {
  #   warning("Cannot estimate standard error with fit.method = none. Forcing estimate.se to FALSE.")
  #   estimate.se = FALSE
  # }
  
  if (scale.X) {
    scl.train = apply(XW_train, 2, sd, na.rm = TRUE)
    is.binary.train = apply(XW_train, 2, function(xx) sum(xx == 0) + sum(xx == 1) == length(xx))
    scl.train[is.binary.train] = 1
    XW_train.scl = scale(XW_train, center = FALSE, scale = scl.train)
    # use the desriptive statistics of train set to scale the test set
    XW_test.scl = scale(XW_test, center = FALSE, scale = scl.train)  
    
    # scl.test = apply(XW_test, 2, sd, na.rm = TRUE)
    # is.binary.test = apply(XW_test, 2, function(xx) sum(xx == 0) + sum(xx == 1) == length(xx))
    # scl.test[is.binary.test] = 1
    # XW_test.scl = scale(XW_test, center = FALSE, scale = scl.test)
  } else {
    XW_train.scl = XW_train
    XW_test.scl = XW_test
  }
  
  # we want ATE for these indices
  target.idx.train = which(W_train %in% target.pop)
  balance.target.train = colMeans(XW_train.scl[target.idx.train,])
  
  target.idx.test = which(W_test %in% target.pop)
  balance.target.test = colMeans(XW_test.scl[target.idx.test,])
  
  if (setequal(target.pop, c(0, 1))) {
    
    est0 = residualBalance.mean(XW_train.scl[W_train==0,], YW_train[W_train==0], XW_test.scl[W_test==0,], YW_test[W_test==0], 
                                             balance.target.train, balance.target.test, binary,
                                             allow.negative.weights, zeta, fit.method, alpha, optimizer=optimizer, bound.gamma=bound.gamma, verbose=verbose)
    est1 = residualBalance.mean(XW_train.scl[W_train==1,], YW_train[W_train==1], XW_test.scl[W_test==1,], YW_test[W_test==1], 
                                             balance.target.train, balance.target.test, binary,
                                             allow.negative.weights, zeta, fit.method, alpha, optimizer=optimizer, bound.gamma=bound.gamma, verbose=verbose)
    
  } else if (setequal(target.pop, c(1))) {
    
    est0 = residualBalance.mean(XW_train.scl[W_train==0,], YW_train[W_train==0], XW_test.scl[W_test==0,], YW_test[W_test==0], 
                                balance.target.train, balance.target.test, binary,
                                allow.negative.weights, zeta, fit.method, alpha, optimizer=optimizer, bound.gamma=bound.gamma, verbose=verbose)
    est1 = c(mean(YW_train[W_train==1]), mean(YW_test[W_test==1]))
    
  } else if (setequal(target.pop, c(0))) {
    
    est0 = c(mean(YW_train[W_train==0]), mean(YW_test[W_test==0]))
    est1 = residualBalance.mean(XW_train.scl[W_train==1,], YW_train[W_train==1], XW_test.scl[W_test==1,], YW_test[W_test==1], 
                                balance.target.train, balance.target.test, binary,
                                allow.negative.weights, zeta, fit.method, alpha, optimizer=optimizer, bound.gamma=bound.gamma, verbose=verbose)
  } else {
    
    stop("Invalid target.pop.")
    
  }
  
  tau.hat.train = est1[1] - est0[1]
  tau.hat.test = est1[2] - est0[2]
  return(c(tau.hat.train, tau.hat.test))
}
