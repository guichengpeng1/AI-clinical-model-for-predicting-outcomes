#!/home/ubuntu/miniconda3/envs/starf/bin/python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv


TIME_COLUMN = "time"
EVENT_COLUMN = "status"
HORIZONS = [36.0, 60.0, 84.0, 120.0]


def structured_target(df):
    return Surv.from_arrays(event=df[EVENT_COLUMN].astype(bool).to_numpy(), time=df[TIME_COLUMN].astype(float).to_numpy())


def harrell_cindex(y, risk_scores):
    return float(concordance_index_censored(y["event"], y["time"], np.asarray(risk_scores, dtype=float))[0])


def ipcw_cindex(y_train, y_eval, risk_scores):
    return float(concordance_index_ipcw(y_train, y_eval, np.asarray(risk_scores, dtype=float))[0])


def brier_at(y_train, eval_df, risk_col):
    if risk_col not in eval_df.columns:
        return {f"test_brier_{int(h)}": np.nan for h in HORIZONS}
    out = {}
    y_eval = structured_target(eval_df)
    for h in HORIZONS:
        horizon_col = f"{risk_col}_{int(h)}"
        if horizon_col not in eval_df.columns:
            out[f"test_brier_{int(h)}"] = np.nan
            continue
        surv_probs = 1.0 - eval_df[horizon_col].to_numpy(dtype=float)
        _, scores = brier_score(y_train, y_eval, surv_probs, np.asarray([h], dtype=float))
        out[f"test_brier_{int(h)}"] = float(scores[0])
    return out


def metric_row(model_name, train_df, val_df, test_df, raw_risk_col):
    y_train = structured_target(train_df)
    y_val = structured_target(val_df)
    y_test = structured_target(test_df)

    row = {
        "model": model_name,
        "val_harrell_cindex": harrell_cindex(y_val, val_df[raw_risk_col]),
        "val_ipcw_cindex": ipcw_cindex(y_train, y_val, val_df[raw_risk_col]),
        "test_harrell_cindex": harrell_cindex(y_test, test_df[raw_risk_col]),
        "test_ipcw_cindex": ipcw_cindex(y_train, y_test, test_df[raw_risk_col]),
    }
    row.update(brier_at(y_train, test_df, raw_risk_col))
    return row


def load_split_manifest(path):
    manifest = pd.read_csv(path)
    return {
        "train": manifest[manifest["split"] == "train"][[TIME_COLUMN, EVENT_COLUMN]].copy(),
        "val": manifest[manifest["split"] == "val"][[TIME_COLUMN, EVENT_COLUMN]].copy(),
        "test": manifest[manifest["split"] == "test"][[TIME_COLUMN, EVENT_COLUMN]].copy(),
    }


def run_r_metrics(manifest_path, r_output_dir):
    splits = load_split_manifest(manifest_path)
    rows = []
    pairs = [
        ("superpc_r", Path(r_output_dir) / "superpc_val_predictions.csv", Path(r_output_dir) / "superpc_test_predictions.csv"),
        (
            "conditional_inference_survival_forest_r",
            Path(r_output_dir) / "cforest_val_predictions.csv",
            Path(r_output_dir) / "cforest_test_predictions.csv",
        ),
    ]
    for model_name, val_path, test_path in pairs:
        if not (val_path.exists() and test_path.exists()):
            continue
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        rows.append(metric_row(model_name, splits["train"], val_df, test_df, "risk"))
    return pd.DataFrame(rows)


def run_python_single_model_metrics(output_dir, model_name):
    manifest_path = Path(output_dir) / "seer_split_manifest.csv"
    val_path = Path(output_dir) / "seer_val_predictions.csv"
    test_path = Path(output_dir) / "seer_test_predictions.csv"
    if not (manifest_path.exists() and val_path.exists() and test_path.exists()):
        return pd.DataFrame()
    splits = load_split_manifest(manifest_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    risk_col = f"risk_{model_name}"
    if risk_col not in val_df.columns or risk_col not in test_df.columns:
        return pd.DataFrame()
    row = metric_row(model_name, splits["train"], val_df, test_df, risk_col)
    return pd.DataFrame([row])


def main():
    parser = argparse.ArgumentParser(description="Compute unified SEER model metrics.")
    parser.add_argument("--base-metrics", default="outputs/ai_clinical_score/seer_model_metrics.csv")
    parser.add_argument("--r-manifest", default="outputs/r_survival_models/r_split_manifest.csv")
    parser.add_argument("--r-output-dir", default="outputs/r_survival_models")
    parser.add_argument("--deephit-output-dir", default="outputs/deephit_full")
    parser.add_argument("--manuscript-survivalquilts-dir", default="outputs/manuscript_survivalquilts")
    parser.add_argument("--output", default="outputs/model_comparison/seer_model_metrics_unified.csv")
    args = parser.parse_args()

    frames = []
    base_path = Path(args.base_metrics)
    if base_path.exists():
        frames.append(pd.read_csv(base_path))
    r_manifest = Path(args.r_manifest)
    if r_manifest.exists():
        r_metrics = run_r_metrics(r_manifest, args.r_output_dir)
        if not r_metrics.empty:
            frames.append(r_metrics)
    deephit_metrics = run_python_single_model_metrics(args.deephit_output_dir, "deephit")
    if not deephit_metrics.empty:
        frames.append(deephit_metrics)
    manuscript_metrics_path = Path(args.manuscript_survivalquilts_dir) / "manuscript_survivalquilts_metrics.csv"
    if manuscript_metrics_path.exists():
        msq = pd.read_csv(manuscript_metrics_path)
        if {"split", "primary_ipcw_cindex"}.issubset(msq.columns):
            val_row = msq[msq["split"] == "val"]
            test_row = msq[msq["split"] == "test"]
            if not val_row.empty and not test_row.empty:
                row = {
                    "model": "manuscript_survivalquilts_9model",
                    "val_harrell_cindex": float(val_row.iloc[0].get("harrell_cindex_60", np.nan)),
                    "val_ipcw_cindex": float(val_row.iloc[0].get("primary_ipcw_cindex", np.nan)),
                    "test_harrell_cindex": float(test_row.iloc[0].get("harrell_cindex_60", np.nan)),
                    "test_ipcw_cindex": float(test_row.iloc[0].get("primary_ipcw_cindex", np.nan)),
                    "test_brier_36": float(test_row.iloc[0].get("brier_36", np.nan)),
                    "test_brier_60": float(test_row.iloc[0].get("brier_60", np.nan)),
                    "test_brier_84": float(test_row.iloc[0].get("brier_84", np.nan)),
                    "test_brier_120": float(test_row.iloc[0].get("brier_120", np.nan)),
                }
                frames.append(pd.DataFrame([row]))

    if not frames:
        raise SystemExit("No metrics found to merge.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "model" in merged.columns:
        merged = merged.drop_duplicates(subset=["model"], keep="last")
        if {"val_ipcw_cindex", "test_ipcw_cindex"}.issubset(merged.columns):
            merged = merged.sort_values(["val_ipcw_cindex", "test_ipcw_cindex"], ascending=False, na_position="last")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(merged.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
