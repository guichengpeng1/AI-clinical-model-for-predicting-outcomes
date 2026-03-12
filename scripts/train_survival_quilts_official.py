#!/home/ubuntu/miniconda3/envs/starf/bin/python
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

from train_ai_clinical_score import (
    EVENT_COLUMN,
    HORIZONS,
    ID_COLUMN,
    PREDICTOR_COLUMNS,
    RANDOM_STATE,
    TIME_COLUMN,
    build_preprocessor,
    prepare_dataframe,
)


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "external" / "mlforhealthlabpub-main" / "alg" / "survivalquilts"
if str(OFFICIAL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DIR))

from class_SurvivalQuilts import SurvivalQuilts as OfficialSurvivalQuilts  # noqa: E402
from class_UnderlyingModels import (  # noqa: E402
    CoxPH,
    CoxPHRidge,
    LogLogistic,
    LogNormal,
    RandomSurvForest,
    Weibull,
)
from utils_eval import calc_metrics  # noqa: E402


class RandomSurvForestPatched(RandomSurvForest):
    def predict(self, X, time_horizons):
        surv = self.model.predict_survival_function(X, return_array=True)
        surv_times = np.asarray(self.model.unique_times_, dtype=float)
        preds = np.zeros((surv.shape[0], len(time_horizons)), dtype=float)

        for t, eval_time in enumerate(time_horizons):
            idx = np.searchsorted(surv_times, float(eval_time), side="left")
            idx = min(idx, len(surv_times) - 1)
            preds[:, t] = 1.0 - surv[:, idx]

        return float(self.direction) * preds


class SurvivalQuiltsSEERAdapter(OfficialSurvivalQuilts):
    def _make_ModelList(self):
        return [
            CoxPH(),
            CoxPHRidge(),
            Weibull(),
            LogNormal(),
            LogLogistic(),
            RandomSurvForestPatched(),
        ]

    def _split_triplet(self, X, T, Y, seed):
        return train_test_split(
            X,
            T,
            Y,
            test_size=0.20,
            random_state=seed + self.SEED,
            stratify=Y.iloc[:, 0],
        )

    def _get_Y_step_pulled(self, W_, X_, T_, Y_, K_step):
        metric_cindex = np.zeros(self.num_cv)
        metric_brier = np.zeros(self.num_cv)

        for cv_idx in range(self.num_cv):
            _, X_va, T_tr, T_va, Y_tr, Y_va = self._split_triplet(X_, T_, Y_, cv_idx)
            pred = self._get_ensemble_prediction(self.CV_pulled_models[cv_idx], W_, X_va, self.time_horizons)

            new_k_step = min(K_step + 1 + self.step_ahead, self.K)
            for k in range(K_step, new_k_step):
                eval_time = self.time_horizons[k]
                tmp_c, tmp_b = calc_metrics(T_tr, Y_tr, T_va, Y_va, pred[:, k], eval_time)

                metric_cindex[cv_idx] += 1.0 / len(self.time_horizons) * tmp_c
                metric_brier[cv_idx] += 1.0 / len(self.time_horizons) * tmp_b
                metric_cindex[cv_idx] += 1.0 / (new_k_step - K_step) * tmp_c
                metric_brier[cv_idx] += 1.0 / (new_k_step - K_step) * tmp_b

        output_cindex = (-metric_cindex.mean(), 1.96 * np.std(metric_cindex) / np.sqrt(self.num_cv))
        output_brier = (metric_brier.mean(), 1.96 * np.std(metric_brier) / np.sqrt(self.num_cv))
        return output_cindex, output_brier

    def _get_models_pulled_CV(self, X, T, Y, seed):
        X_tr, X_va, T_tr, T_va, Y_tr, Y_va = self._split_triplet(X, T, Y, seed)
        pulled_models = self._get_trained_models(X_tr, T_tr, Y_tr)
        metric_cindex = np.zeros((self.M, self.K), dtype=float)
        metric_brier = np.zeros((self.M, self.K), dtype=float)

        for m, model in enumerate(pulled_models):
            pred = model.predict(X_va, self.time_horizons)
            for t, eval_time in enumerate(self.time_horizons):
                tmp_c, tmp_b = calc_metrics(T_tr, Y_tr, T_va, Y_va, pred[:, t], eval_time)
                metric_cindex[m, t] = tmp_c
                metric_brier[m, t] = tmp_b

        return pulled_models, metric_cindex, metric_brier


def split_dataframe(df):
    X = df[PREDICTOR_COLUMNS]
    y = df[EVENT_COLUMN]
    X_train, X_temp, idx_train, idx_temp = train_test_split(
        X,
        df.index.to_numpy(),
        test_size=0.36,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, idx_val, idx_test = train_test_split(
        X_temp,
        idx_temp,
        test_size=(20.0 / 36.0),
        random_state=RANDOM_STATE,
        stratify=df.loc[idx_temp, EVENT_COLUMN],
    )
    return idx_train, idx_val, idx_test


def encode_split_frames(df, idx_train, idx_val, idx_test):
    preprocessor = build_preprocessor()
    X_train_raw = df.loc[idx_train, PREDICTOR_COLUMNS]
    X_val_raw = df.loc[idx_val, PREDICTOR_COLUMNS]
    X_test_raw = df.loc[idx_test, PREDICTOR_COLUMNS]

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = [str(x) for x in preprocessor.get_feature_names_out()]

    return (
        preprocessor,
        pd.DataFrame(X_train, index=idx_train, columns=feature_names),
        pd.DataFrame(X_val, index=idx_val, columns=feature_names),
        pd.DataFrame(X_test, index=idx_test, columns=feature_names),
    )


def structured_target(frame):
    return Surv.from_arrays(
        event=frame[EVENT_COLUMN].astype(bool).to_numpy(),
        time=frame[TIME_COLUMN].astype(float).to_numpy(),
    )


def evaluate_split(y_train, y_eval, pred_frame):
    row = {}
    ipcw_values = []
    for horizon in HORIZONS:
        risk = pred_frame[f"risk_{int(horizon)}"].to_numpy(dtype=float)
        surv = 1.0 - np.clip(risk, 0.0, 1.0)
        row[f"harrell_cindex_{int(horizon)}"] = float(
            concordance_index_censored(y_eval["event"], y_eval["time"], risk)[0]
        )
        row[f"ipcw_cindex_{int(horizon)}"] = float(concordance_index_ipcw(y_train, y_eval, risk, horizon)[0])
        row[f"brier_{int(horizon)}"] = float(
            brier_score(y_train, y_eval, surv, np.asarray([horizon], dtype=float))[1][0]
        )
        ipcw_values.append(row[f"ipcw_cindex_{int(horizon)}"])
    row["ipcw_cindex_mean"] = float(np.mean(ipcw_values))
    row["primary_risk_horizon"] = 60
    row["primary_ipcw_cindex"] = row["ipcw_cindex_60"]
    return row


def build_prediction_frame(df, idx, pred_matrix):
    out = df.loc[idx, [ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    for i, horizon in enumerate(HORIZONS):
        out[f"risk_{int(horizon)}"] = pred_matrix[:, i]
    out["risk_primary_60"] = out["risk_60"]
    return out


def train_and_evaluate(df, output_dir, args):
    idx_train, idx_val, idx_test = split_dataframe(df)
    preprocessor, X_train, X_val, X_test = encode_split_frames(df, idx_train, idx_val, idx_test)
    T_train = df.loc[idx_train, [TIME_COLUMN]].copy()
    Y_train = df.loc[idx_train, [EVENT_COLUMN]].copy()

    model = SurvivalQuiltsSEERAdapter(
        K=args.K,
        num_bo=args.num_bo,
        num_outer=args.num_outer,
        num_cv=args.num_cv,
        step_ahead=args.step_ahead,
    )
    model.train(X_train, T_train, Y_train)

    val_pred = model.predict(X_val, eval_time_horizons=HORIZONS)
    test_pred = model.predict(X_test, eval_time_horizons=HORIZONS)
    all_pred = model.predict(
        pd.concat([X_train, X_val, X_test], axis=0).loc[df.index],
        eval_time_horizons=HORIZONS,
    )

    val_pred_df = build_prediction_frame(df, idx_val, val_pred)
    test_pred_df = build_prediction_frame(df, idx_test, test_pred)
    all_pred_df = build_prediction_frame(df, df.index, all_pred)
    split_df = df[[ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    split_df["split"] = "train"
    split_df.loc[idx_val, "split"] = "val"
    split_df.loc[idx_test, "split"] = "test"

    y_train = structured_target(df.loc[idx_train])
    y_val = structured_target(df.loc[idx_val])
    y_test = structured_target(df.loc[idx_test])
    metrics = pd.DataFrame(
        [
            {"split": "val", **evaluate_split(y_train, y_val, val_pred_df)},
            {"split": "test", **evaluate_split(y_train, y_test, test_pred_df)},
        ]
    )

    summary = {
        "official_source_dir": str(OFFICIAL_DIR),
        "official_base_models": model.model_names,
        "split": {
            "n_total": int(len(df)),
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
        },
        "training_config": {
            "K": args.K,
            "num_bo": args.num_bo,
            "num_outer": args.num_outer,
            "num_cv": args.num_cv,
            "step_ahead": args.step_ahead,
        },
        "primary_metric": "ipcw_cindex_60",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "survivalquilts_metrics.csv", index=False)
    val_pred_df.to_csv(output_dir / "survivalquilts_val_predictions.csv", index=False)
    test_pred_df.to_csv(output_dir / "survivalquilts_test_predictions.csv", index=False)
    all_pred_df.to_csv(output_dir / "survivalquilts_all_predictions.csv", index=False)
    split_df.to_csv(output_dir / "seer_split_manifest.csv", index=False)
    with open(output_dir / "survivalquilts_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "predictors": PREDICTOR_COLUMNS,
            "horizons": HORIZONS,
            "primary_horizon": 60,
            "official_source_dir": str(OFFICIAL_DIR),
        },
        output_dir / "survivalquilts_bundle.joblib",
    )
    return metrics, summary


def main():
    parser = argparse.ArgumentParser(description="Adapt official Survival Quilts to the local SEER AI-clinical-score data.")
    parser.add_argument("--input", default="AIdata/SEER.csv")
    parser.add_argument("--output-dir", default="outputs/survivalquilts_official")
    parser.add_argument("--sample-n", type=int, default=0, help="Optional smoke subset size.")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--num-bo", type=int, default=8)
    parser.add_argument("--num-outer", type=int, default=2)
    parser.add_argument("--num-cv", type=int, default=3)
    parser.add_argument("--step-ahead", type=int, default=2)
    args = parser.parse_args()

    df = prepare_dataframe(Path(args.input))
    if args.sample_n:
        df = df.sample(n=min(args.sample_n, len(df)), random_state=RANDOM_STATE).sort_index()

    metrics, summary = train_and_evaluate(df, Path(args.output_dir), args)
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
