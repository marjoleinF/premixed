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
#' @param cluster optional character string supplying the name of the cluster 
#' indicator. If specified, \code{formula} should not involve random effects 
#' (e.g., \code{y ~ x1+ x2 + x3}). If \code{cluster} is specified, random 
#' effects will not be estimated during tree induction. This will substantially 
#' speed up computations, but may yield a less accurate model, depending on the 
#' magnitude of the random effects.
#' @param conv.thresh numeric vector of length 1, specifies the convergence
#' criterion. The algorithm converges if the maximum absolute difference in 
#' random-effects predictions from one iteration is smaller than 
#' \code{conv.thresh}.
#' @param penalty.par.val as usual.
#' @param learnrate as usual.
#' @param use.grad as usual.
#' @param family as usual. Note: should be a character vector!
#' @param ridge.ranef logical vector of length 1. Should random effects be
#' estimated through a ridge regression? If set to \code{TRUE}, random effects 
#' will be estimated through fitting a ridge regression model using function 
#' \code{\link[glmnet]{cv.glmnet}}. If set to \code{FALSE}, random
#' effects will be estimated through fitting a mixed-effects regression model
#' using function \code{\link[lme4]{lmer}} or \code{\link[lme4]{glmer}}. 
#' @param max.iter numeric vector of length 1. Maximum number of iterations
#' performed to re-estimate fixed and random effects parameters. 
#' @param ... further arguments to be passed to \code{\link[pre]{pre}}.
#' @description Experimental function for fitting mixed-effects prediction rule
#' ensembles. Estimates a random intercept in addition to a prediction rule 
#' ensemble. This allows for analysing datasets with a clustered or multilevel
#' structure, or longitudinal datasets. Experimental, so use at own risk. 
#' 
#' Function premixed() allows for taking into account a random intercept in I) 
#' rule induction and/or II) coefficient estimation. To take into account the 
#' random intercept in both rule induction and coefficient estimation, see 
#' Example 1 below. To take into account the random intercept only in 
#' coefficient estimation, see Example 2 below. Alternatively, it has been 
#' suggested that random effects do not need to be taken into account
#' explicitly but only through employing a blocked bootstrap or subampling 
#' approach, see Exemple 3 below. Note that approaches / examples 1 and 2 can 
#' be combined with the third approach / example 3. See Example 4 below. 
#' 
#' Note that random intercept-only models are currently supported. That is, 
#' random slopes can currently not be specified.
#'  
#' @return An object of class 'premixed'.
#' @examples \donttest{
#' 
#' ## Example 1: Take into account clustered structure in rule induction
#' ## as well as coeficient estimation: 
#' set.seed(42)
#' airq <- airquality[complete.cases(airquality),]
#' airq.ens1 <- premixed(Ozone ~ 1 | Month | Solar.R + Wind + Temp + Day, data = airq, ntrees = 10)
#' airq.ens1
#' 
#' 
#' ## Example 2: Take into account clustered stucture in coefficient estimation
#' ## only:
#' set.seed(42)
#' airq <- airquality[complete.cases(airquality),]
#' airq.ens1 <- premixed(Ozone ~ 1 | Month | Solar.R + Wind + Temp + Day, data = airq, ntrees = 10)
#' airq.ens1
#' 
#' 
#' 
#' 
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
#' 
#' 
#' ## Employ ctree with blocked bootstrapsampling for rule inducation,
#' ## and include random effects only in estimation of the final ensemble:
#' bb_sampfunc <- function(cluster = airq$Month) {
#'   result <- c()
#'   for(i in sample(unique(cluster), replace = TRUE)) {
#'     result <- c(result, which(cluster == i))
#'   }
#'   result
#' }
#' set.seed(42)
#' airq.ens5 <- premixed(Ozone ~ Solar.R + Wind + Temp + Day, cluster = "Month", 
#'   data = airq, sampfrac = bb_sampfunc)
#' airq.ens5
#' 
#' }   
#' @export
#' 
#' 
# library(pre)
# library(glmnet)
# library(lme4)
# formula <- Ozone ~ Solar.R + Wind + Temp + Day 
# cluster <- "Month"
# data <- airquality[complete.cases(airquality),]
# penalty.par.val = "lambda.min"
# learnrate=0
# use.grad=F
# conv.thresh=.01
# family="gaussian"
# ridge.ranef = TRUE
# max.iter=1000

premixed <- function(formula, cluster = NULL, data, penalty.par.val = "lambda.min", 
                        learnrate = 0, use.grad = FALSE, conv.thresh = .01, 
                        family = "gaussian", ridge.ranef = FALSE, max.iter = 1000, ...) {
  cat("estimating rules...\n")
  ## TODO: implement different types:
  # 1) Take into account ranefs in tree estimation: a) glmertree, or b) ctree with blocked bootstrap sampling
  # 2) Take into account ranefs in coef estimation
  ## TODO: implement blocked bootstrap sampling
  pre_mod <- pre(formula = formula, data = data, learnrate = learnrate,
                 use.grad = use.grad, family = family)#, ...)
  if (is.null(cluster)) {
    cluster_name <- as.character(formula[[3]][[2]][[3]])
    if (ridge.ranef) {
      cluster <- data.frame(factor(data[, cluster_name]))
      names(cluster) <- cluster_name
    } else {
      y_name <- all.vars(formula, max.names = 1)
      glmer_data <- data.frame(data[y_name], cluster = data[,cluster_name])
    }
  } else {
    cluster_name <- cluster
    if (ridge.ranef) {
      cluster <- data.frame(factor(data[,cluster_name]))
      names(cluster) <- cluster_name
    } else {
      y_name <- all.vars(formula, max.names = 1)
      glmer_data <- data.frame(data[y_name], cluster = data[,cluster_name])
    }
  }

  if (ridge.ranef) {
    names(cluster) <- cluster_name
    ranef_des_mat <- as.matrix(with(cluster, model.matrix(formula(paste("~", cluster_name, "-1")))))
    fixef_preds <- pre:::predict.pre(pre_mod, penalty.par.val = penalty.par.val)
  } else {
    glmer_data$fixef_preds <- pre:::predict.pre(pre_mod, penalty.par.val = penalty.par.val)
    glmer_f <- formula(paste0(y_name, " ~ -1 + (1|cluster) + offset(fixef_preds)"))
    y_name <- all.vars(formula, max.names = 1)
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
                             newoffset = 0, type = "link")
      ranef1[[iteration]] <- coef(ranef, s = "lambda.min")
    } else {
      if (family == "gaussian") { ## fit lmer:
        ranef <- lmer(glmer_f, data = glmer_data)
        ranef1[[iteration]] <- ranef(ranef)[[1]]
        glmer_data$fixef_preds <- 0
        ranef_preds <- predict(ranef, newdata = glmer_data)
      } else { ## otherwise, fit glmer:
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
                             newoffset = 0, type = "link")
    } else {
      glmer_data$fixef_preds <- predict(fixef, newx = pre_mod$modmat, 
                                        s = "lambda.min", newoffset = 0, 
                                        type = "link")
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
  print(x$pre, ...)
}


#' Return predicted values based on a mixed-effects prediction rule ensemble.
#' 
#' \code{predict.premixed} returns predicted values based on the estimated
#' fixed effects of a a mixed-effects prediction rule ensemble (random 
#' effects are NOT included in the predictions (yet)). 
#' 
#' @param object an object of class 'premixed'.
#' @param offset Offset value to be used in generating predictions. Note that
#' the predictions are returned for the fixed effects only.
#' @param ... further arguments to be passed to \code{\link[pre]{predict.pre}}
#' 
#' @examples \donttest{
#' ## Employ glmertree for rule induction and iterate between lasso for
#' ## estimating fixed effects and ridge for estimating random effects:
#' set.seed(42)
#' airq <- airquality[complete.cases(airquality),]
#' airq.ens1 <- premixed(Ozone ~ 1 | Month | Solar.R + Wind + Temp + Day, data = airq, ntrees = 10)
#' predict(airq.ens1, newdata = airq)
#' }
predict.premixed <- function(object, offset = 0, ...) {
  predict(object$pre, newoffset = offset, ...)
}


#' Return estimated fixed-effects coefficients from a mixed-effects prediction
#' rule ensemble
#' 
#' \code{coef.premixed} returns the estimated fixed-effects coefficients from 
#' a mixed-effects prediction rule ensemble
#' 
#' @param object an object of class 'premixed'.
#' @param ... further arguments to be passed to \code{\link[pre]{coef.pre}}.
#' 
coef.premixed <- function(object, ...) {
  coef(object$pre, ...)
}
  

#' Return predicted random-effects coefficients from a mixed-effects prediction 
#' rule ensemble
#' 
#' \code{ranef.premixed} returns the predicted random-effects coefficients from a 
#' mixed-effects prediction rule ensemble.
#' 
#' @param object an object of class'premixed'.
#' @param ... currently not used.
ranef.premixed <- function(object, ...) {
  ## Add printed statement on how coefficients were estimated / predicted
  object$ranef1
}



#' Calculates complexity of a prediction rule ensemble.
#' 
#' \code{complexity} returns the complexity (total number of variables in the 
#' ensemble, number of terms and mean number of variables per term) of a
#' prediction rule ensembles (i.e., an object of class 'pre' or 'premixed').
#' 
#' @param object an object of class 'pre' or premixed'.
#' @param penalty.par.val As usual.
#' @param ... not currently used.
#' 
#' @examples \donttest{
#' set.seed(42)
#' airq <- airquality[complete.cases(airquality),]
#' airq.ens1 <- premixed(Ozone ~ 1 | Month | Solar.R + Wind + Temp + Day, data = airq, ntrees = 10)
#' complexity(airq.ens1)
#' }
#' 
#' @return Returns a vector with the total number of variables in the ensemble,
#' the total number of terms (i.e., baselearners with a non-zero coefficient),
#' and the mean number of variables per term) in the ensemble.
complexity <- function(object, penalty.par.val = "lambda.1se", ...) {
  
  ## Preliminaries:
  if (class(object) == "premixed") {
    object <- object$pre
  }
  if (!class(object) == "pre") {
    stop("Argument object should sprecify an object of class 'pre'.")
  }
  
  ## Get non-zero coefficients:
  coefs <- coef(object)
  nonzerocoefs <- coefs[coefs$coefficient != 0,]
  ## Count average number of variables in terms:
  mean_no_of_vars <- mean(lengths(
    regmatches(nonzerocoefs$description[-1], 
               gregexpr(" & ", nonzerocoefs$description[-1])))) + 1
  ## Count the number of variables:
  imps <- importance(object, penalty.par.val = penalty.par.val, 
                     plot = FALSE)
  ## Count the number of terms:
  if (penalty.par.val == "lambda.1se") {
    lambda_ind <- which(object$glmnet.fit$lambda == object$glmnet.fit$lambda.1se)
  }
  if (penalty.par.val == "lambda.min") {
    lambda_ind <- which(object$glmnet.fit$lambda == object$glmnet.fit$lambda.min)
  }
  if (is.numeric(penalty.par.val)) {
    lambda_ind <- which(abs(object$glmnet.fit$lambda - penalty.par.val) == 
                          min(abs(object$glmnet.fit$lambda - penalty.par.val)))
  }
  nterms <- object$glmnet.fit$nzero[lambda_ind][[1]]
  nvars <- nrow(imps$varimp)
  
  ## Return resultS:
  c(nvars = nvars, nterms = nterms, mean_no_of_vars = mean_no_of_vars)
}