#!/home/ubuntu/miniconda3/envs/r4.3/bin/Rscript

suppressPackageStartupMessages({
  library(survival)
  library(partykit)
})

args <- commandArgs(trailingOnly = TRUE)
artifact_dir <- if (length(args) >= 1) args[[1]] else stop("missing artifact_dir")
input_csv <- if (length(args) >= 2) args[[2]] else stop("missing input_csv")
output_csv <- if (length(args) >= 3) args[[3]] else stop("missing output_csv")

km_surv_at <- function(survfit_obj, horizon) {
  ss <- summary(survfit_obj, times = horizon, extend = TRUE)
  if (length(ss$surv) == 0) {
    return(NA_real_)
  }
  as.numeric(ss$surv[[1]])
}

baseline_surv_at <- function(basehaz_df, horizon) {
  idx <- max(which(basehaz_df$time <= horizon))
  if (!is.finite(idx)) {
    return(1.0)
  }
  exp(-basehaz_df$hazard[idx])
}

predict_superpc_bundle <- function(bundle, df) {
  X <- as.matrix(df[, bundle$predictors, drop = FALSE])
  score_x <- X[, bundle$selected, drop = FALSE]
  pca_scores <- predict(bundle$pca_fit, newdata = score_x)
  if (is.null(dim(pca_scores))) {
    pc1 <- as.numeric(pca_scores)
  } else {
    pc1 <- as.numeric(pca_scores[, 1])
  }
  lp <- as.numeric(predict(bundle$cox_fit, newdata = data.frame(score = pc1), type = "lp"))
  out <- data.frame(superpc_risk = lp)
  for (h in bundle$horizons) {
    s0 <- baseline_surv_at(bundle$basehaz_df, h)
    out[[paste0("superpc_risk_", h)]] <- 1 - s0 ^ exp(lp)
  }
  out
}

predict_prob_chunked <- function(model, newdata, chunk_size = 2000L) {
  n <- nrow(newdata)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  out <- vector("list", length(idx))
  for (i in seq_along(idx)) {
    out[[i]] <- predict(model, newdata = newdata[idx[[i]], , drop = FALSE], type = "prob")
  }
  unlist(out, recursive = FALSE, use.names = FALSE)
}

predict_cforest_bundle <- function(bundle, df) {
  surv_list <- predict_prob_chunked(bundle$cf_fit, df[, bundle$predictors, drop = FALSE])
  risk120 <- vapply(surv_list, function(sf) 1 - km_surv_at(sf, 120), numeric(1))
  out <- data.frame(cforest_risk = risk120)
  for (h in bundle$horizons) {
    out[[paste0("cforest_risk_", h)]] <- vapply(surv_list, function(sf) 1 - km_surv_at(sf, h), numeric(1))
  }
  out
}

superpc_bundle <- readRDS(file.path(artifact_dir, "superpc_model.rds"))
cforest_bundle <- readRDS(file.path(artifact_dir, "cforest_model.rds"))
input_df <- read.csv(input_csv, check.names = FALSE)

required_cols <- unique(c(superpc_bundle$predictors, cforest_bundle$predictors))
missing_cols <- setdiff(required_cols, names(input_df))
if (length(missing_cols) > 0) {
  stop(sprintf("missing predictor columns: %s", paste(missing_cols, collapse = ", ")))
}

for (col in required_cols) {
  input_df[[col]] <- as.numeric(input_df[[col]])
}

superpc_pred <- predict_superpc_bundle(superpc_bundle, input_df)
cforest_pred <- predict_cforest_bundle(cforest_bundle, input_df)

out <- cbind(input_df, superpc_pred, cforest_pred)
write.csv(out, output_csv, row.names = FALSE)
