#' Fitting a prediction rule ensemble with clustered data
#' 
#' \code{premixed} is a wrapper for fitting a PRe with clustered data
#'  
#' @param formula a formula with three-part right-hand side, like 
#' \code{y ~ 1 | cluster | x1 + x2 + x3}; or with one-part right hand side, like
#' \code{y ~ x1 + x2 + x3}. In the latter case, the cluster indicator must
#' be specified through the \code{cluster} argument.
#' #' @param data a dataframe containing the variables in the model
#' @param data dataframe containing the variables specified in \code{formula}.
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
#' @param family as usual. Note: should be a character vector!
#' @param ridge.ranef logical vector of length 1. Should random effects be
#' estimated using a ridge penalty paramater? If set to \code{FALSE}, random
#' effects will be estimated through fitting a mixed-effects regression model
#' using function \code{\link[lme4]{lmer}} or \code{\link[lme4]{glmer}}.
#' @param max.iter numeric vector of length 1. Maximum number of iterations
#' performed to re-estimate fixed and random effects parameters. 
#' @param ... further arguments to be passed to \code{\link[pre]{pre}}.
#' @description Experimental function, use at own risk. Estimates rules by 
#' using function glmertree(). Estimates random- and fixed-effects coefficients
#' by iterating between lasso estimation of the fixed-effects (i.e., rules 
#' and/or linear terms) and ridge estimation of the random-effects predictions.
#' The model involves a random intercept only.
#' @return An object of class 'premixed'.
#' @examples \donttest{
#' data(DepressionDemo, package = "glmertree")
#' 
#' 
#' ## Employ glmertree for rule induction and iterate between lasso for
#' ## estimating fixed effects and ridge for estimating random effects:
#' set.seed(42)
#' airq <- airquality[complete.cases(airquality),]
#' airq.ens1 <- premixed(Ozone ~ 1 | Month | Solar.R + Wind + Temp + Day, data = airq, ntrees = 10)
#' airq.ens1
#' 
#' 
#' ## Employ glmertree for rule induction and iterate between lasso for
#' ## estimating fixed effects and glmer for estimating random effects:
#' set.seed(42)
#' airq.ens2 <- premixed(Ozone ~ 1 | Month | Solar.R + Wind + Temp + Day,
#'                       data = airq, ridge.ranef = FALSE, ntrees = 10)
#' airq.ens2
#' 
#' 
#' ## Employ ctree with blocked bootstrap sampling for rule induction:
#' ##
#' ## First create a sampling function that bootstrap samples whole clusters:
#' bb_sampfunc <- function(cluster = airq$Month) {
#'   result <- c()
#'   for(i in sample(unique(cluster), replace = TRUE)) {
#'     result <- c(result, which(cluster == i))
#'   }
#'   result
#' }
#' ## And then fit the PRE:
#' library(pre)
#' set.seed(42)
#' airq.ens3 <- pre(Ozone ~ ., data = airq, sampfrac = bb_sampfunc)
#' airq.ens3
#' 
#' 
#' ## Employ ctree with blocked subsampling for rule induction:
#' ##
#' ## First create a function that creates subsamples containing ~75% of the clusters: 
#' ss_sampfunc <- function(cluster = airq$Month, sampfrac = .75) {
#'   result <- c()
#'   n_clusters <- round(length(unique(cluster)) * sampfrac)
#'   for(i in sample(unique(cluster), size = n_clusters, replace = FALSE)) {
#'     result <- c(result, which(cluster == i))
#'   }
#'   result
#' }
#' ## And then fit the PRE:
#' library(pre)
#' set.seed(42)
#' airq.ens4 <- pre(Ozone ~ ., data = airq, sampfrac = ss_sampfunc)
#' airq.ens4
#' }   
#' @export
premixed <- function(formula, cluster = NULL, data, penalty.par.val = "lambda.min", 
                        learnrate = 0, use.grad = FALSE, conv.thresh = .01, 
                        family = "gaussian", ridge.ranef = TRUE, max.iter = 1000, ...) {
  cat("estimating rules...\n")
  ## TODO: implement different types:
  # 1) Take into account ranefs in tree estimation: a) glmertree, or b) ctree with blocked bootstrap sampling
  # 2) Take into account ranefs in coef estimation
  ## TODO: implement blocked bootstrap sampling 
  pre_mod <- pre(formula = formula, data = data, learnrate = learnrate,
                 use.grad = use.grad, family = family, ...)
  if (is.null(cluster)) {
    cluster_name <- as.character(formula[[3]][[2]][[3]])
    if (ridge.ranef) {
      cluster <- data.frame(factor(pre_mod$data[, cluster_name]))
    } else {
      y_name <- all.vars(formula, max.names = 1)
      glmer_data <- data.frame(data[y_name], cluster = data[,cluster_name])
    }
  } else {
    if (ridge.ranef) {
      cluster <- data.frame(factor(cluster))
    } else {
      y_name <- all.vars(formula, max.names = 1)
      glmer_data <- data.frame(data[y_name], cluster = cluster)
    }
    cluster_name <- "cluster"
  }

  if (ridge.ranef) {
    names(cluster) <- cluster_name
    ranef_des_mat <- as.matrix(with(cluster, model.matrix(formula(paste("~", cluster_name, "-1")))))
    fixef_preds <- pre:::predict.pre(pre_mod, penalty.par.val = penalty.par.val)
  } else {
    glmer_data$fixef_preds <- pre:::predict.pre(pre_mod, penalty.par.val = penalty.par.val)
  }
  iteration <- 0
  dif <- conv.thresh + 1
  ranef1 <- list()
  fixef1 <- list()
  
  ## TODO: pre-create sparse matrix for ranef and fixef
  ## TODO: ridge regression lambda parameter should be equal to something of the error variance. 
  ## TODO: allow for specifying whether trees should be grown with glmertree, or without ranef estimation
  
  cat("estimating coefficients...\n")
  while (dif > conv.thresh && iteration < max.iter) {
    iteration <- iteration + 1
    ## Step 1: estimate random effects, given fixed-effects coefficients:
    if (ridge.ranef) {
      ranef <- cv.glmnet(x = ranef_des_mat, y = pre_mod$data[, pre_mod$y_name], 
                         alpha = 0, standardize = FALSE, offset = fixef_preds, 
                         intercept = FALSE, family = family)
      ranef_preds <- predict(ranef, newx = ranef_des_mat, s = "lambda.min", 
                             offset = 0, type = "link")
      ranef1[[iteration]] <- coef(ranef, s = "lambda.min")
    } else {
      y_name <- all.vars(formula, max.names = 1)
      if (family == "gaussian") { ## fit lmer:
        glmer_f <- formula(paste0(y_name, " ~ -1 + (1|cluster) + offset(fixef_preds)"))
        ranef <- lmer(glmer_f, data = glmer_data)
        ranef1[[iteration]] <- ranef(ranef)[[1]]
        glmer_data$fixef_preds <- 0
        ranef_preds <- predict(ranef, newdata = glmer_data)
      } else { ## otherwise, fit glmer:
        glmer_f <- formula(paste0(y_name, " ~ -1 + (1|cluster) + offset(fixef_preds)"))
        ranef <- glmer(glmer_f, data = glmer_data, family = family)
        ranef1[[iteration]] <- ranef(ranef)[[1]]
        glmer_data$fixef_preds <- 0        
        ranef_preds <- predict(ranef, newdata = glmer_data, type = "link")
      }
    }
    ## Step 2: estimate fixed effects, given random-effects coefficients:
    fixef <- cv.glmnet(x = pre_mod$modmat, y = pre_mod$data[, pre_mod$y_name], 
                       alpha = 1, standardize = FALSE, offset = ranef_preds,
                       family = family)
    if (ridge.ranef) {
      fixef_preds <- predict(fixef, newx = pre_mod$modmat, s = "lambda.min", 
                            offset = 0, type = "link")
    } else {
      glmer_data$fixef_preds <- predict(fixef, newx = pre_mod$modmat, s = "lambda.min", 
                             offset = 0, type = "link")
    }
    fixef1[[iteration]] <- coef(fixef, s = "lambda.min")
    if (iteration > 1) {
      dif <- max(abs(ranef1[[iteration]] - ranef1[[iteration-1]]))
    }
  }
  if (iteration == max.iter) {
    warning("Estimation of random and fixed effects stopped, but did not converge before reaching the maximum number of iterations.")
  }
  pre_mod$glmnet.fit <- fixef
  result <- list(pre = pre_mod,
                 iterations = iteration,
                 fixef = fixef1,
                 ranef = ranef1,
                 call = match.call())
  class(result) <- "premixed"
  return(result)
}

print.premixed <- function(x, ...) {
  if ( (is.null(x$call$max.iter) && x$iterations == 1000) || 
      (!is.null(x$call$max.iter) && x$iterations == x$call$max.iter) ) {
    cat("\nEstimation of random and fixed effects parameters took", x$iterations, "iterations and did not converge.\n")    
  } else {
    cat("\nEstimation of random and fixed effects parameters converged after", x$iterations, "iterations.\n")
  }
  pre:::print.pre(x$pre, ...)
}

#predict.cluster.pre <- function()
#coef.cluster.pre <- function()
#ranef.cluster.pre <- function()