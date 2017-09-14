#' Fitting a prediction rule ensemble with clustered data
#' 
#' \code{premixed} is a wrapper for fitting a PRe with clustered data
#'  
#' @param formula a Formula object with three-part right-hand side like 
#' y ~ 1 | cluster | x1 + x2 + x3
#' @param data a dataframe containing the variables in the model
#' @param cluster optional string supplying the name of the cluster indicator. If specified,
#' \code{formula} should not involve random effects (e.g., 
#' \code{y ~ x1+ x2 + x3}). If \code{cluster} is specified, random effects will
#' not be estimated during tree induction. This will substantially speed up 
#' computations, but may yield a less accurate model, depending on the magnitude
#' of the random effects.
#' @param conv.thresh numeric vector of length 1, specifies the convergence
#' criterion. The algorithm converges if the maximum absolute difference in 
#' the random-effects coefficients from one iteration is smaller than 
#' \code{conv.thresh}.
#' @param penalty.par.val as usual.
#' @param learnrate as usual.
#' @param use.grad as usual.
#' @param family as usual. 
#' @param max.iter numeric vector of length 1. Maximum number of iterations
#' performed to re-estimate fixed and random effects parameters. 
#' @param ... further arguments to be passed to \code{\link[pre]{pre}}.
#' @description Experimental function, use at own risk. Estimates rules by 
#' using function glmertree(). Estimates random- and fixed-effects coefficients
#' by iterating between lasso estimation of the fixed-effects (i.e., rules 
#' and/or linear terms) and ridge estimation of the random-effects predictions.
#' The model involves a random intercept only.
#' @return An object of class 'premixed'.
#' @examples \donttest{set.seed(42)
#` airq.ens <- cluster.pre(Ozone ~ 1 | Month | Solar.R + Wind + Temp + Day, data = airq)
#' }
#' @export
premixed <- function(formula, cluster = NULL, data, penalty.par.val = "lambda.min", 
                        learnrate = 0, use.grad = FALSE, conv.thresh = .5, 
                        family = "gaussian", max.iter = 1000, ...) {
  cat("estimating rules...\n")
  pre_mod <- pre(formula = formula, data = data, learnrate = learnrate,
                 use.grad = use.grad, family = family, ...)
  if (nrow(pre_mod$rules) == 1) {
    return(NULL)
    cat("ended!!!!")
  }
  if (is.null(cluster)) {
    cluster_name <- as.character(formula[[3]][[2]][[3]])
    cluster <- data.frame(factor(pre_mod$data[, cluster_name]))
  } else {
    cluster <- data.frame(factor(cluster))
    cluster_name <- "cluster"
  }
  names(cluster) <- cluster_name
  ranef_des_mat <- as.matrix(with(cluster, model.matrix(formula(paste("~", cluster_name, "-1")))))
  fixef_preds <- pre:::predict.pre(pre_mod, penalty.par.val = penalty.par.val)
  iteration <- 0
  dif <- conv.thresh + 1
  ranef1 <- list()
  fixef1 <- list()
  ## TODO: pre-create sparse matrix for ranef and fixef
  ## TODO: ridge regression lambda parameter should be equal to something of the error variance. 
  ## TODO: allow for specifying whether trees should be grown with glmertree, or without ranef estimation
  
  ## TODO: Only continue when rules have been derived
  cat("estimating coefficients...\n")
  while (dif > conv.thresh && iteration < max.iter) {
    iteration <- iteration + 1
    ## Step 1: estimate random effects, given fixed-effects coefficients:
    ranef <- glmnet:::cv.glmnet(x = ranef_des_mat, y = pre_mod$data[, pre_mod$y_name], 
                       alpha = 0, standardize = FALSE, offset = fixef_preds, 
                       intercept = FALSE, family = family)
    ranef_preds <- glmnet:::predict.cv.glmnet(ranef, newx = ranef_des_mat, s = "lambda.min", 
                           offset = 0, type = "link")
    ##
    ranef1[[iteration]] <- glmnet:::coef.glmnet(ranef, s = "lambda.min")
    ## Step 2: estimate fixed effects, given random-effects coefficients:
    fixef <- glmnet:::cv.glmnet(x = pre_mod$modmat, y = pre_mod$data[, pre_mod$y_name], 
                       alpha = 1, standardize = FALSE, offset = ranef_preds,
                       family = family)
    fixef_preds <- glmnet:::predict.cv.glmnet(fixef, newx = pre_mod$modmat, s = "lambda.min", 
                           offset = 0, type = "link")
    fixef1[[iteration]] <- glmnet:::coef.glmnet(fixef, s = "lambda.min")
    if (iteration > 1) {
      dif <- max(abs(ranef1[[iteration]] - ranef1[[iteration-1]]))
    }
  }
  pre_mod$glmnet.fit <- fixef
  result <- list(pre = pre_mod,
                 iterations = iteration,
                 fixef = fixef1,
                 ranef = ranef1)
  class(result) <- "premixed"
  return(result)
}

print.premixed <- function(x, ...) {
  cat("\nEstimation of random and fixed effects parameters converged after", x$iterations, "iterations.\n")
  pre:::print.pre(x$pre, ...)
}

#predict.cluster.pre <- function()
#coef.cluster.pre <- function()
#ranef.cluster.pre <- function()