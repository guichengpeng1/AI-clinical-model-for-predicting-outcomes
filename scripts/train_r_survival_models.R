#!/home/ubuntu/miniconda3/envs/r4.3/bin/Rscript

suppressPackageStartupMessages({
  library(survival)
  library(superpc)
  library(partykit)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "AIdata/SEER.csv"
output_dir <- if (length(args) >= 2) args[[2]] else "outputs/r_survival_models"
split_manifest_path <- if (length(args) >= 3 && nzchar(args[[3]])) args[[3]] else ""

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

predictors <- c(
  "Age",
  "Sex",
  "Tumor Size Summary (2016+)",
  "Grade Pathological (2018+)",
  "Tsgemerge1",
  "Nstagemerge1",
  "Derived EOD 2018 M (2018+)",
  "Race recode (W, B, AI, API)",
  "Histologic Type ICD-O-3(1chRCC,2pRCC,3ccRCC)"
)
horizons <- c(36, 60, 84, 120)

read_data <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  keep <- c("Patient ID", "time", "status", predictors)
  df <- df[, keep]
  df <- df[complete.cases(df), ]
  df <- df[df$time > 0 & df$Age >= 18 & df$Age <= 90, ]
  for (col in predictors) {
    df[[col]] <- as.numeric(df[[col]])
  }
  df$status <- as.integer(df$status)
  df
}

stratified_split <- function(df, train_frac = 0.64, val_frac = 0.16, seed = 20260309) {
  set.seed(seed)
  idx_train <- c()
  idx_val <- c()
  idx_test <- c()
  for (event_value in sort(unique(df$status))) {
    idx <- which(df$status == event_value)
    idx <- sample(idx, length(idx))
    n <- length(idx)
    n_train <- floor(n * train_frac)
    n_val <- floor(n * val_frac)
    idx_train <- c(idx_train, idx[seq_len(n_train)])
    idx_val <- c(idx_val, idx[seq(n_train + 1, n_train + n_val)])
    idx_test <- c(idx_test, idx[seq(n_train + n_val + 1, n)])
  }
  list(
    train = df[sort(idx_train), ],
    val = df[sort(idx_val), ],
    test = df[sort(idx_test), ]
  )
}

manifest_split <- function(df, manifest_path) {
  manifest <- read.csv(manifest_path, check.names = FALSE)
  keys <- c("Patient ID", "time", "status")
  if (!all(c(keys, "split") %in% names(manifest))) {
    stop("split manifest must contain Patient ID, time, status, split")
  }
  joined <- merge(
    df,
    manifest[, c(keys, "split")],
    by = keys,
    all.x = TRUE,
    sort = FALSE
  )
  if (any(is.na(joined$split))) {
    stop("split manifest did not match all rows in the filtered SEER dataset")
  }
  list(
    train = joined[joined$split == "train", c(keys, predictors)],
    val = joined[joined$split == "val", c(keys, predictors)],
    test = joined[joined$split == "test", c(keys, predictors)]
  )
}

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

fit_superpc <- function(train_df, val_df, test_df) {
  train_list <- list(
    x = t(as.matrix(train_df[, predictors])),
    y = train_df$time,
    censoring.status = train_df$status,
    featurenames = predictors
  )
  val_list <- list(
    x = t(as.matrix(val_df[, predictors])),
    y = val_df$time,
    censoring.status = val_df$status,
    featurenames = predictors
  )
  test_list <- list(
    x = t(as.matrix(test_df[, predictors])),
    y = test_df$time,
    censoring.status = test_df$status,
    featurenames = predictors
  )

  sp_fit <- superpc.train(train_list, type = "survival")
  score_abs <- sort(abs(sp_fit$feature.scores), decreasing = TRUE)
  keep_n <- min(length(score_abs), max(5, ceiling(length(score_abs) * 0.5)))
  best_threshold <- score_abs[keep_n]
  selected <- which(abs(sp_fit$feature.scores) >= best_threshold)
  if (length(selected) < 1) {
    selected <- order(abs(sp_fit$feature.scores), decreasing = TRUE)[1:min(3, length(sp_fit$feature.scores))]
  }

  train_x <- t(train_list$x[selected, , drop = FALSE])
  val_x <- t(val_list$x[selected, , drop = FALSE])
  test_x <- t(test_list$x[selected, , drop = FALSE])

  pca_fit <- prcomp(train_x, center = TRUE, scale. = FALSE)
  train_score <- as.numeric(predict(pca_fit, newdata = train_x)[, 1])
  val_score <- as.numeric(predict(pca_fit, newdata = val_x)[, 1])
  test_score <- as.numeric(predict(pca_fit, newdata = test_x)[, 1])

  cox_df <- data.frame(time = train_df$time, status = train_df$status, score = train_score)
  cox_fit <- coxph(Surv(time, status) ~ score, data = cox_df, x = TRUE)
  basehaz_df <- basehaz(cox_fit, centered = FALSE)

  train_lp <- as.numeric(predict(cox_fit, newdata = data.frame(score = train_score), type = "lp"))
  val_lp <- as.numeric(predict(cox_fit, newdata = data.frame(score = val_score), type = "lp"))
  test_lp <- as.numeric(predict(cox_fit, newdata = data.frame(score = test_score), type = "lp"))

  train_out <- data.frame(id = train_df$`Patient ID`, time = train_df$time, status = train_df$status, risk = train_lp)
  val_out <- data.frame(id = val_df$`Patient ID`, time = val_df$time, status = val_df$status, risk = val_lp)
  test_out <- data.frame(id = test_df$`Patient ID`, time = test_df$time, status = test_df$status, risk = test_lp)
  for (h in horizons) {
    s0 <- baseline_surv_at(basehaz_df, h)
    train_out[[paste0("risk_", h)]] <- 1 - s0 ^ exp(train_lp)
    val_out[[paste0("risk_", h)]] <- 1 - s0 ^ exp(val_lp)
    test_out[[paste0("risk_", h)]] <- 1 - s0 ^ exp(test_lp)
  }

  metrics <- data.frame(
    model = "superpc_r",
    val_harrell_cindex = NA_real_,
    test_harrell_cindex = NA_real_,
    threshold = best_threshold
  )
  artifact <- list(
    kind = "superpc_r",
    predictors = predictors,
    horizons = horizons,
    selected = selected,
    pca_fit = pca_fit,
    cox_fit = cox_fit,
    basehaz_df = basehaz_df
  )
  list(metrics = metrics, train = train_out, val = val_out, test = test_out, artifact = artifact)
}

fit_cforest <- function(train_df, val_df, test_df) {
  train_use <- train_df[, c("time", "status", predictors)]
  val_use <- val_df[, c("time", "status", predictors)]
  test_use <- test_df[, c("time", "status", predictors)]
  formula <- as.formula(paste("Surv(time, status) ~", paste(sprintf("`%s`", predictors), collapse = " + ")))

  cf_fit <- cforest(
    formula,
    data = train_use,
    ntree = 200L,
    mtry = ceiling(sqrt(length(predictors))),
    trace = FALSE,
    control = ctree_control(teststat = "quad", testtype = "Univ", mincriterion = 0, saveinfo = FALSE)
  )

  predict_prob_chunked <- function(model, newdata, chunk_size = 2000L) {
    n <- nrow(newdata)
    idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
    out <- vector("list", length(idx))
    for (i in seq_along(idx)) {
      out[[i]] <- predict(model, newdata = newdata[idx[[i]], , drop = FALSE], type = "prob")
    }
    unlist(out, recursive = FALSE, use.names = FALSE)
  }

  surv_to_row <- function(sf) {
    out <- c()
    for (h in horizons) {
      out <- c(out, 1 - km_surv_at(sf, h))
    }
    names(out) <- paste0("risk_", horizons)
    out
  }

  train_prob <- predict_prob_chunked(cf_fit, train_use)
  val_prob <- predict_prob_chunked(cf_fit, val_use)
  test_prob <- predict_prob_chunked(cf_fit, test_use)

  train_risk120 <- vapply(train_prob, function(sf) 1 - km_surv_at(sf, 120), numeric(1))
  val_risk120 <- vapply(val_prob, function(sf) 1 - km_surv_at(sf, 120), numeric(1))
  test_risk120 <- vapply(test_prob, function(sf) 1 - km_surv_at(sf, 120), numeric(1))

  train_out <- data.frame(id = train_df$`Patient ID`, time = train_df$time, status = train_df$status, risk = train_risk120)
  val_out <- data.frame(id = val_df$`Patient ID`, time = val_df$time, status = val_df$status, risk = val_risk120)
  test_out <- data.frame(id = test_df$`Patient ID`, time = test_df$time, status = test_df$status, risk = test_risk120)
  train_extra <- t(vapply(train_prob, surv_to_row, numeric(length(horizons))))
  val_extra <- t(vapply(val_prob, surv_to_row, numeric(length(horizons))))
  test_extra <- t(vapply(test_prob, surv_to_row, numeric(length(horizons))))
  train_out <- cbind(train_out, as.data.frame(train_extra))
  val_out <- cbind(val_out, as.data.frame(val_extra))
  test_out <- cbind(test_out, as.data.frame(test_extra))

  metrics <- data.frame(
    model = "conditional_inference_survival_forest_r",
    val_harrell_cindex = NA_real_,
    test_harrell_cindex = NA_real_,
    threshold = NA_real_
  )
  artifact <- list(
    kind = "conditional_inference_survival_forest_r",
    predictors = predictors,
    horizons = horizons,
    cf_fit = cf_fit
  )
  list(metrics = metrics, train = train_out, val = val_out, test = test_out, artifact = artifact)
}

df <- read_data(input_path)
splits <- if (nzchar(split_manifest_path)) manifest_split(df, split_manifest_path) else stratified_split(df)

superpc_res <- fit_superpc(splits$train, splits$val, splits$test)
cforest_res <- fit_cforest(splits$train, splits$val, splits$test)

metrics <- rbind(superpc_res$metrics, cforest_res$metrics)
write.csv(metrics, file.path(output_dir, "r_model_metrics.csv"), row.names = FALSE)
write.csv(superpc_res$train, file.path(output_dir, "superpc_train_predictions.csv"), row.names = FALSE)
write.csv(superpc_res$val, file.path(output_dir, "superpc_val_predictions.csv"), row.names = FALSE)
write.csv(superpc_res$test, file.path(output_dir, "superpc_test_predictions.csv"), row.names = FALSE)
saveRDS(superpc_res$artifact, file.path(output_dir, "superpc_model.rds"))
write.csv(cforest_res$train, file.path(output_dir, "cforest_train_predictions.csv"), row.names = FALSE)
write.csv(cforest_res$val, file.path(output_dir, "cforest_val_predictions.csv"), row.names = FALSE)
write.csv(cforest_res$test, file.path(output_dir, "cforest_test_predictions.csv"), row.names = FALSE)
saveRDS(cforest_res$artifact, file.path(output_dir, "cforest_model.rds"))

cat("Finished R survival models\n")
print(metrics)
