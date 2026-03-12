#!/home/ubuntu/miniconda3/envs/r4.3/bin/Rscript

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "AIdata/SEER.csv"
output_path <- if (length(args) >= 2) args[[2]] else "outputs/r_survival_models/r_split_manifest.csv"

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

read_data <- function(path) {
  df <- read.csv(path, check.names = FALSE)
  keep <- c("Patient ID", "time", "status", predictors)
  df <- df[, keep]
  df <- df[complete.cases(df), ]
  df <- df[df$time > 0 & df$Age >= 18 & df$Age <= 90, ]
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
  manifest <- data.frame(
    `Patient ID` = df$`Patient ID`,
    time = df$time,
    status = df$status,
    split = "train"
  )
  manifest$split[sort(idx_val)] <- "val"
  manifest$split[sort(idx_test)] <- "test"
  manifest
}

df <- read_data(input_path)
manifest <- stratified_split(df)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.csv(manifest, output_path, row.names = FALSE)
cat("Wrote", output_path, "\n")
