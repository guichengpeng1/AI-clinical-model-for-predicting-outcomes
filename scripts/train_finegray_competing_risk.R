#!/home/ubuntu/miniconda3/envs/r4.3/bin/Rscript

suppressPackageStartupMessages({
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "AIdata/TCGA_CPTAC1_revised.csv"
output_dir <- if (length(args) >= 2) args[[2]] else "outputs/competing_risk_finegray"
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

detect_columns <- function(df) {
  id_col <- if ("Patient ID" %in% names(df)) "Patient ID" else if ("ID" %in% names(df)) "ID" else if ("id" %in% names(df)) "id" else stop("No ID column found")
  overall_time <- "time"
  overall_status <- "status"
  if (!all(c(overall_time, overall_status) %in% names(df))) {
    stop("Expected time/status columns for competing risk branch")
  }
  if (all(c("CSS.time", "CSS.status") %in% names(df))) {
    return(list(id = id_col, overall_time = overall_time, overall_status = overall_status, cause_time = "CSS.time", cause_status = "CSS.status"))
  }
  if (all(c("css_time", "cssStatus1") %in% names(df))) {
    return(list(id = id_col, overall_time = overall_time, overall_status = overall_status, cause_time = "css_time", cause_status = "cssStatus1"))
  }
  if (all(c("css_time", "cssStatus") %in% names(df))) {
    return(list(id = id_col, overall_time = overall_time, overall_status = overall_status, cause_time = "css_time", cause_status = "cssStatus"))
  }
  stop("No cause-specific time/status columns found")
}

pick_time_scale <- function(overall_time, cause_time, cause_status) {
  idx <- which(!is.na(cause_time) & cause_status == 1 & cause_time > 0 & overall_time > 0)
  if (length(idx) < 5) {
    return(1.0)
  }
  candidates <- c(1.0, 1 / 30.4375, 1 / 365.25, 12.0)
  scores <- sapply(candidates, function(scale) median(abs(overall_time[idx] - cause_time[idx] * scale), na.rm = TRUE))
  candidates[which.min(scores)]
}

read_competing_risk_data <- function(path) {
  raw <- read.csv(path, check.names = FALSE)
  cols <- detect_columns(raw)
  keep <- unique(c(cols$id, cols$overall_time, cols$overall_status, cols$cause_time, cols$cause_status, predictors))
  df <- raw[, keep]
  names(df)[names(df) == cols$id] <- "id"
  names(df)[names(df) == cols$overall_time] <- "overall_time"
  names(df)[names(df) == cols$overall_status] <- "overall_status"
  names(df)[names(df) == cols$cause_time] <- "cause_time"
  names(df)[names(df) == cols$cause_status] <- "cause_status"

  df <- df[complete.cases(df[, c("id", "overall_time", "overall_status", predictors)]), ]
  df <- df[df$overall_time > 0 & df$Age >= 18 & df$Age <= 90, ]
  for (col in predictors) {
    df[[col]] <- as.numeric(df[[col]])
  }
  df$overall_status <- as.integer(df$overall_status)
  df$cause_status <- as.integer(ifelse(is.na(df$cause_status), 0L, df$cause_status))
  df$cause_time <- as.numeric(ifelse(is.na(df$cause_time), 0, df$cause_time))

  time_scale <- pick_time_scale(df$overall_time, df$cause_time, df$cause_status)
  converted_cause_time <- df$cause_time * time_scale
  df$fg_event <- ifelse(df$cause_status == 1, 1L, ifelse(df$overall_status == 1, 2L, 0L))
  df$fg_time <- ifelse(df$fg_event == 1 & converted_cause_time > 0, converted_cause_time, df$overall_time)
  df <- df[!is.na(df$fg_event) & !is.na(df$fg_time) & df$fg_time > 0, ]

  list(data = df, meta = list(id_col = cols$id, cause_time_col = cols$cause_time, cause_status_col = cols$cause_status, time_scale = time_scale))
}

split_df <- function(df, split_manifest_path = "") {
  if (nzchar(split_manifest_path)) {
    manifest <- read.csv(split_manifest_path, check.names = FALSE)
    keys <- intersect(c("Patient ID", "ID", "id"), names(manifest))
    if (length(keys) < 1 || !"split" %in% names(manifest)) {
      stop("Split manifest must contain an ID column and split")
    }
    key <- keys[[1]]
    names(manifest)[names(manifest) == key] <- "id"
    merged <- merge(df, manifest[, c("id", "split")], by = "id", all.x = TRUE, sort = FALSE)
    if (any(is.na(merged$split))) {
      stop("Split manifest did not match all competing-risk rows")
    }
    return(list(train = merged[merged$split == "train", ], val = merged[merged$split == "val", ], test = merged[merged$split == "test", ]))
  }

  set.seed(20260309)
  idx_train <- c()
  idx_val <- c()
  idx_test <- c()
  for (event_value in sort(unique(df$fg_event))) {
    idx <- which(df$fg_event == event_value)
    idx <- sample(idx, length(idx))
    n <- length(idx)
    n_train <- floor(n * 0.64)
    n_val <- floor(n * 0.16)
    idx_train <- c(idx_train, idx[seq_len(n_train)])
    idx_val <- c(idx_val, idx[seq(n_train + 1, n_train + n_val)])
    idx_test <- c(idx_test, idx[seq(n_train + n_val + 1, n)])
  }
  list(train = df[sort(idx_train), ], val = df[sort(idx_val), ], test = df[sort(idx_test), ])
}

baseline_surv_at <- function(basehaz_df, horizon) {
  idx <- max(which(basehaz_df$time <= horizon))
  if (!is.finite(idx)) {
    return(1.0)
  }
  exp(-basehaz_df$hazard[idx])
}

fit_finegray <- function(train_df, val_df, test_df) {
  formula <- as.formula(
    paste("Surv(fg_time, factor(fg_event)) ~", paste(sprintf("`%s`", predictors), collapse = " + "))
  )
  fg_train <- finegray(formula, data = train_df, etype = "1")
  cox_formula <- as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", paste(sprintf("`%s`", predictors), collapse = " + ")))
  fg_fit <- coxph(cox_formula, data = fg_train, weights = fgwt, ties = "breslow", x = TRUE)
  basehaz_df <- basehaz(fg_fit, centered = FALSE)

  to_output <- function(df) {
    lp <- as.numeric(predict(fg_fit, newdata = df, type = "lp"))
    out <- data.frame(id = df$id, time = df$fg_time, event_type = df$fg_event, risk = lp)
    for (h in horizons) {
      s0 <- baseline_surv_at(basehaz_df, h)
      out[[paste0("risk_", h)]] <- 1 - s0 ^ exp(lp)
    }
    out
  }

  list(
    train = to_output(train_df),
    val = to_output(val_df),
    test = to_output(test_df)
  )
}

loaded <- read_competing_risk_data(input_path)
df <- loaded$data
splits <- split_df(df, split_manifest_path)
fg_res <- fit_finegray(splits$train, splits$val, splits$test)

summary_df <- data.frame(
  input = input_path,
  id_col = loaded$meta$id_col,
  cause_time_col = loaded$meta$cause_time_col,
  cause_status_col = loaded$meta$cause_status_col,
  cause_time_scale_to_overall = loaded$meta$time_scale,
  n_total = nrow(df),
  n_train = nrow(splits$train),
  n_val = nrow(splits$val),
  n_test = nrow(splits$test),
  n_event_interest = sum(df$fg_event == 1, na.rm = TRUE),
  n_competing_event = sum(df$fg_event == 2, na.rm = TRUE),
  n_censored = sum(df$fg_event == 0, na.rm = TRUE)
)

write.csv(summary_df, file.path(output_dir, "finegray_summary.csv"), row.names = FALSE)
write.csv(fg_res$train, file.path(output_dir, "finegray_train_predictions.csv"), row.names = FALSE)
write.csv(fg_res$val, file.path(output_dir, "finegray_val_predictions.csv"), row.names = FALSE)
write.csv(fg_res$test, file.path(output_dir, "finegray_test_predictions.csv"), row.names = FALSE)

cat("Finished Fine-Gray competing-risk branch\n")
print(summary_df)
