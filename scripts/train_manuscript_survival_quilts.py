#!/home/ubuntu/miniconda3/envs/starf/bin/python
import argparse
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

from train_ai_clinical_score import (
    EVENT_COLUMN,
    HORIZONS,
    ID_COLUMN,
    PREDICTOR_COLUMNS,
    TIME_COLUMN,
    RANDOM_STATE,
    create_split_manifest,
    fit_and_evaluate,
    prepare_dataframe,
)


PYTHON_MODELS = [
    "cox_ph",
    "lasso",
    "elastic_net",
    "survival_svm",
    "gbm",
    "xgboost_survival",
    "lightgbm_survival",
    "random_survival_forest",
    "deephit",
]
R_MODELS = [
    "superpc_r",
    "conditional_inference_survival_forest_r",
]
MANUSCRIPT_MODELS = PYTHON_MODELS + R_MODELS
R_SCRIPT = Path("/home/ubuntu/miniconda3/envs/r4.3/bin/Rscript")


def structured_target(df):
    return Surv.from_arrays(
        event=df[EVENT_COLUMN].astype(bool).to_numpy(),
        time=df[TIME_COLUMN].astype(float).to_numpy(),
    )


def materialize_filtered_input(input_path, output_dir, sample_n=0):
    df = prepare_dataframe(input_path)
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=RANDOM_STATE).sort_index()
    staged_path = output_dir / "seer_filtered_input.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(staged_path, index=False)
    return df.reset_index(drop=True), staged_path


def ensure_base_predictions(df, staged_input_path, split_manifest_path, python_out_dir, r_out_dir):
    fit_and_evaluate(
        df,
        python_out_dir,
        staged_input_path,
        model_names=PYTHON_MODELS,
        split_manifest_path=split_manifest_path,
    )

    subprocess.run(
        [str(R_SCRIPT), "scripts/train_r_survival_models.R", str(staged_input_path), str(r_out_dir), str(split_manifest_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def rename_r_columns(df, model_name):
    df = df.copy()
    df = df.rename(columns={"id": ID_COLUMN})
    renamed = {
        "risk": f"risk_{model_name}",
        "risk_36": f"risk_{model_name}_36",
        "risk_60": f"risk_{model_name}_60",
        "risk_84": f"risk_{model_name}_84",
        "risk_120": f"risk_{model_name}_120",
    }
    for src, dst in renamed.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
    return df


def load_split_predictions(output_dir, split_name):
    return pd.read_csv(Path(output_dir) / f"seer_{split_name}_predictions.csv")


def load_r_split_predictions(output_dir, prefix, split_name, model_name):
    path = Path(output_dir) / f"{prefix}_{split_name}_predictions.csv"
    return rename_r_columns(pd.read_csv(path), model_name)


def merge_prediction_frames(base_df, extras):
    merged = base_df.copy()
    keys = [ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]
    for extra in extras:
        merged = merged.merge(extra, on=keys, how="left", validate="one_to_one")
    return merged


def normalize_from_train(train_values, eval_values):
    lo = float(np.nanmin(train_values))
    hi = float(np.nanmax(train_values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(eval_values, 0.5, dtype=float)
    scaled = (eval_values - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def model_horizon_values(train_df, eval_df, model_name, horizon):
    horizon_col = f"risk_{model_name}_{int(horizon)}"
    raw_col = f"risk_{model_name}"
    if horizon_col in train_df.columns and horizon_col in eval_df.columns:
        return eval_df[horizon_col].to_numpy(dtype=float)
    if raw_col not in train_df.columns or raw_col not in eval_df.columns:
        raise KeyError(f"missing columns for model {model_name} at horizon {horizon}")
    return normalize_from_train(
        train_df[raw_col].to_numpy(dtype=float),
        eval_df[raw_col].to_numpy(dtype=float),
    )


def build_horizon_matrix(train_df, eval_df, horizon):
    cols = [model_horizon_values(train_df, eval_df, model_name, horizon) for model_name in MANUSCRIPT_MODELS]
    return np.column_stack(cols)


def score_weights_ipcw(y_train, y_eval, matrix, weights, horizon):
    risk = np.asarray(matrix @ weights, dtype=float)
    return float(concordance_index_ipcw(y_train, y_eval, risk, horizon)[0])


def optimize_horizon_weights_random_search(y_train, y_val, train_df, val_df, horizon, random_draws=4000):
    matrix_val = build_horizon_matrix(train_df, val_df, horizon)
    rng = np.random.default_rng(RANDOM_STATE + int(horizon))
    candidates = [np.eye(len(MANUSCRIPT_MODELS))[i] for i in range(len(MANUSCRIPT_MODELS))]
    candidates.append(np.full(len(MANUSCRIPT_MODELS), 1.0 / len(MANUSCRIPT_MODELS)))
    candidates.extend(rng.dirichlet(np.ones(len(MANUSCRIPT_MODELS)), size=random_draws))

    best_score = -np.inf
    best_weights = None
    for weights in candidates:
        score = score_weights_ipcw(y_train, y_val, matrix_val, weights, horizon)
        if score > best_score:
            best_score = score
            best_weights = np.asarray(weights, dtype=float)

    focused = rng.dirichlet(1.0 + 50.0 * best_weights, size=max(1000, random_draws // 2))
    for weights in focused:
        score = score_weights_ipcw(y_train, y_val, matrix_val, weights, horizon)
        if score > best_score:
            best_score = score
            best_weights = np.asarray(weights, dtype=float)

    return best_weights / best_weights.sum(), best_score


def optimize_horizon_weights_super_learner(y_train, y_val, train_df, val_df, horizon):
    matrix_val = build_horizon_matrix(train_df, val_df, horizon)
    surv_truth = structured_target(val_df)

    def objective(weights):
        risk = np.clip(matrix_val @ weights, 1e-6, 1.0 - 1e-6)
        surv = 1.0 - risk
        return float(brier_score(y_train, surv_truth, surv, np.asarray([horizon], dtype=float))[1][0])

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0) for _ in MANUSCRIPT_MODELS]
    starts = [np.full(len(MANUSCRIPT_MODELS), 1.0 / len(MANUSCRIPT_MODELS))]
    starts.extend(np.eye(len(MANUSCRIPT_MODELS)))

    best_weights = starts[0]
    best_obj = np.inf
    for start in starts:
        result = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        weights = np.asarray(result.x if result.success else start, dtype=float)
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            weights = np.full(len(MANUSCRIPT_MODELS), 1.0 / len(MANUSCRIPT_MODELS))
        weights = weights / weights.sum()
        obj = objective(weights)
        if obj < best_obj:
            best_obj = obj
            best_weights = weights

    return best_weights, score_weights_ipcw(y_train, y_val, matrix_val, best_weights, horizon)


def optimize_horizon_weights(y_train, y_val, train_df, val_df, horizon, random_draws=4000, method="super_learner"):
    if method == "random_search":
        return optimize_horizon_weights_random_search(y_train, y_val, train_df, val_df, horizon, random_draws=random_draws)
    if method == "super_learner":
        return optimize_horizon_weights_super_learner(y_train, y_val, train_df, val_df, horizon)
    raise ValueError(f"unknown ensemble method: {method}")


def prediction_frame(df, train_df, eval_df, weights_by_horizon):
    out = eval_df[[ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    for horizon in HORIZONS:
        matrix = build_horizon_matrix(train_df, eval_df, horizon)
        weights = weights_by_horizon[int(horizon)]
        out[f"risk_{int(horizon)}"] = matrix @ weights
    out["risk_primary_60"] = out["risk_60"]
    return out


def evaluate_predictions(y_train, y_eval, pred_df):
    row = {}
    for horizon in HORIZONS:
        risk = pred_df[f"risk_{int(horizon)}"].to_numpy(dtype=float)
        surv = 1.0 - np.clip(risk, 0.0, 1.0)
        row[f"harrell_cindex_{int(horizon)}"] = float(concordance_index_censored(y_eval["event"], y_eval["time"], risk)[0])
        row[f"ipcw_cindex_{int(horizon)}"] = float(concordance_index_ipcw(y_train, y_eval, risk, horizon)[0])
        row[f"brier_{int(horizon)}"] = float(
            brier_score(y_train, y_eval, surv, np.asarray([horizon], dtype=float))[1][0]
        )
    row["ipcw_cindex_mean"] = float(np.mean([row[f"ipcw_cindex_{int(h)}"] for h in HORIZONS]))
    row["primary_ipcw_cindex"] = row["ipcw_cindex_60"]
    return row


def weights_table(weights_by_horizon, val_scores):
    rows = []
    for horizon in HORIZONS:
        weights = weights_by_horizon[int(horizon)]
        for model_name, weight in zip(MANUSCRIPT_MODELS, weights):
            rows.append(
                {
                    "horizon": int(horizon),
                    "model": model_name,
                    "weight": float(weight),
                    "val_ipcw_cindex": float(val_scores[int(horizon)]),
                }
            )
    return pd.DataFrame(rows)


def collect_raw_normalization(train_df):
    normalization = {}
    for model_name in MANUSCRIPT_MODELS:
        horizon_col = f"risk_{model_name}_{int(HORIZONS[0])}"
        raw_col = f"risk_{model_name}"
        if horizon_col not in train_df.columns and raw_col in train_df.columns:
            values = train_df[raw_col].to_numpy(dtype=float)
            normalization[model_name] = {
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
    return normalization


def main():
    parser = argparse.ArgumentParser(description="Train manuscript-aligned Survival Quilts over 11 base models.")
    parser.add_argument("--input", default="AIdata/SEER.csv")
    parser.add_argument("--output-dir", default="outputs/manuscript_survivalquilts")
    parser.add_argument("--sample-n", type=int, default=0)
    parser.add_argument("--random-draws", type=int, default=4000)
    parser.add_argument(
        "--ensemble-method",
        default="super_learner",
        choices=["super_learner", "random_search"],
        help="Meta-learner used to combine base survival models.",
    )
    parser.add_argument(
        "--skip-base-training",
        action="store_true",
        help="Reuse existing python_base_models and r_base_models under output-dir.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    python_out_dir = output_dir / "python_base_models"
    r_out_dir = output_dir / "r_base_models"
    if args.skip_base_training:
        staged_input_path = output_dir / "seer_filtered_input.csv"
        split_manifest_path = output_dir / "seer_split_manifest.csv"
        if not staged_input_path.exists():
            raise FileNotFoundError(f"missing staged input: {staged_input_path}")
        if not split_manifest_path.exists():
            raise FileNotFoundError(f"missing split manifest: {split_manifest_path}")
        df = pd.read_csv(staged_input_path)
    else:
        df, staged_input_path = materialize_filtered_input(Path(args.input), output_dir, sample_n=args.sample_n)
        split_manifest = create_split_manifest(df)
        split_manifest_path = output_dir / "seer_split_manifest.csv"
        split_manifest.to_csv(split_manifest_path, index=False)
        ensure_base_predictions(df, staged_input_path, split_manifest_path, python_out_dir, r_out_dir)

    train_df = merge_prediction_frames(
        load_split_predictions(python_out_dir, "train"),
        [
            load_r_split_predictions(r_out_dir, "superpc", "train", "superpc_r"),
            load_r_split_predictions(r_out_dir, "cforest", "train", "conditional_inference_survival_forest_r"),
        ],
    )
    val_df = merge_prediction_frames(
        load_split_predictions(python_out_dir, "val"),
        [
            load_r_split_predictions(r_out_dir, "superpc", "val", "superpc_r"),
            load_r_split_predictions(r_out_dir, "cforest", "val", "conditional_inference_survival_forest_r"),
        ],
    )
    test_df = merge_prediction_frames(
        load_split_predictions(python_out_dir, "test"),
        [
            load_r_split_predictions(r_out_dir, "superpc", "test", "superpc_r"),
            load_r_split_predictions(r_out_dir, "cforest", "test", "conditional_inference_survival_forest_r"),
        ],
    )

    y_train = structured_target(train_df)
    y_val = structured_target(val_df)
    y_test = structured_target(test_df)

    weights_by_horizon = {}
    val_scores = {}
    for horizon in HORIZONS:
        weights, score = optimize_horizon_weights(
            y_train,
            y_val,
            train_df,
            val_df,
            horizon,
            random_draws=args.random_draws,
            method=args.ensemble_method,
        )
        weights_by_horizon[int(horizon)] = weights
        val_scores[int(horizon)] = score

    train_ensemble = prediction_frame(df, train_df, train_df, weights_by_horizon)
    val_ensemble = prediction_frame(df, train_df, val_df, weights_by_horizon)
    test_ensemble = prediction_frame(df, train_df, test_df, weights_by_horizon)

    metrics = pd.DataFrame(
        [
            {"split": "train", **evaluate_predictions(y_train, y_train, train_ensemble)},
            {"split": "val", **evaluate_predictions(y_train, y_val, val_ensemble)},
            {"split": "test", **evaluate_predictions(y_train, y_test, test_ensemble)},
        ]
    )
    weights_df = weights_table(weights_by_horizon, val_scores)
    raw_normalization = collect_raw_normalization(train_df)

    summary = {
        "input": str(args.input),
        "staged_input": str(staged_input_path),
        "models": MANUSCRIPT_MODELS,
        "horizons": [int(h) for h in HORIZONS],
        "random_draws": int(args.random_draws),
        "ensemble_method": str(args.ensemble_method),
        "sample_n": int(args.sample_n),
        "skip_base_training": bool(args.skip_base_training),
        "split": {
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
        "primary_metric": "ipcw_cindex_60",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    train_ensemble.to_csv(output_dir / "manuscript_survivalquilts_train_predictions.csv", index=False)
    val_ensemble.to_csv(output_dir / "manuscript_survivalquilts_val_predictions.csv", index=False)
    test_ensemble.to_csv(output_dir / "manuscript_survivalquilts_test_predictions.csv", index=False)
    weights_df.to_csv(output_dir / "manuscript_survivalquilts_weights.csv", index=False)
    metrics.to_csv(output_dir / "manuscript_survivalquilts_metrics.csv", index=False)
    with open(output_dir / "manuscript_survivalquilts_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    joblib.dump(
        {
            "bundle_type": "manuscript_survivalquilts_ensemble",
            "weights_by_horizon": weights_by_horizon,
            "models": MANUSCRIPT_MODELS,
            "predictors": PREDICTOR_COLUMNS,
            "horizons": [int(h) for h in HORIZONS],
            "ensemble_method": str(args.ensemble_method),
            "raw_normalization": raw_normalization,
            "split_manifest": str(split_manifest_path),
            "python_base_dir": str(python_out_dir),
            "python_models_path": str(python_out_dir / "python_models.joblib"),
            "r_base_dir": str(r_out_dir),
            "r_prediction_script": str(Path(__file__).resolve().parents[1] / "scripts" / "predict_r_survival_models.R"),
        },
        output_dir / "manuscript_survivalquilts_bundle.joblib",
    )

    print(metrics.to_string(index=False))
    print(weights_df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
