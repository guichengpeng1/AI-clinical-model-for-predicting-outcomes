#!/home/ubuntu/miniconda3/envs/starf/bin/python
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
import torchtuples as tt
import xgboost as xgb
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


PREDICTOR_COLUMNS = [
    "Age",
    "Sex",
    "Tumor Size Summary (2016+)",
    "Grade Pathological (2018+)",
    "Tsgemerge1",
    "Nstagemerge1",
    "Derived EOD 2018 M (2018+)",
    "Race recode (W, B, AI, API)",
    "Histologic Type ICD-O-3(1chRCC,2pRCC,3ccRCC)",
]

TIME_COLUMN = "time"
EVENT_COLUMN = "status"
ID_COLUMN = "Patient ID"
HORIZONS = [36.0, 60.0, 84.0, 120.0]
RANDOM_STATE = 20260309


class CoxCalibratedRiskModel:
    def _fit_calibrator(self, raw_risk, y):
        X_cal = np.asarray(raw_risk, dtype=float).reshape(-1, 1)
        self.calibrator = CoxPHSurvivalAnalysis(alpha=1e-4)
        self.calibrator.fit(X_cal, y)

    def predict_survival_function(self, X):
        raw_risk = np.asarray(self.predict(X), dtype=float).reshape(-1, 1)
        return self.calibrator.predict_survival_function(raw_risk)


class XGBoostSurvivalWrapper(CoxCalibratedRiskModel):
    def __init__(
        self,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = xgb.XGBRegressor(
            objective="survival:cox",
            eval_metric="cox-nloglik",
            tree_method="hist",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    def _encode_target(self, y):
        times = np.asarray(y["time"], dtype=float)
        events = np.asarray(y["event"], dtype=bool)
        return np.where(events, times, -times)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.model.fit(X, self._encode_target(y), verbose=False)
        self._fit_calibrator(self.model.predict(X), y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.asarray(self.model.predict(X), dtype=float)


class LightGBMSurvivalWrapper(CoxCalibratedRiskModel):
    def __init__(
        self,
        num_boost_round=300,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
    ):
        self.num_boost_round = num_boost_round
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.model = None
        self._event_field = "_event_indicator"

    def _cox_objective(self, preds, train_data):
        times = np.asarray(train_data.get_label(), dtype=float)
        events = np.asarray(train_data.get_weight(), dtype=float)
        order = np.argsort(times, kind="mergesort")
        exp_shifted = np.exp(preds[order] - np.max(preds[order]))
        risk_sums = np.cumsum(exp_shifted[::-1])[::-1] + 1e-12
        inv_risk = events[order] / risk_sums
        inv_risk_sq = events[order] / np.square(risk_sums)
        accum_1 = np.cumsum(inv_risk)
        accum_2 = np.cumsum(inv_risk_sq)

        grad_sorted = -events[order] + exp_shifted * accum_1
        hess_sorted = np.maximum(exp_shifted * accum_1 - np.square(exp_shifted) * accum_2, 1e-6)

        grad = np.empty_like(grad_sorted)
        hess = np.empty_like(hess_sorted)
        grad[order] = grad_sorted
        hess[order] = hess_sorted
        return grad, hess

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        times = np.asarray(y["time"], dtype=float)
        events = np.asarray(y["event"], dtype=float)
        train_set = lgb.Dataset(X, label=times, weight=events, free_raw_data=False)
        params = {
            "objective": self._cox_objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "num_threads": 1,
        }
        self.model = lgb.train(params=params, train_set=train_set, num_boost_round=self.num_boost_round)
        self._fit_calibrator(self.predict(X), y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.asarray(self.model.predict(X), dtype=float)


class SuperPCSimplified:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        self.model = CoxPHSurvivalAnalysis(alpha=1e-4)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        Xp = self.pca.fit_transform(Xs)
        self.model.fit(Xp, y)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        return self.model.predict(Xp)

    def predict_survival_function(self, X):
        Xs = self.scaler.transform(X)
        Xp = self.pca.transform(Xs)
        return self.model.predict_survival_function(Xp)


class DeepHitWrapper:
    def __init__(self, num_durations=32, epochs=8, batch_size=512, lr=1e-2):
        self.num_durations = num_durations
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.labtrans = LabTransDiscreteTime(num_durations)
        self.model = None
        self.duration_index = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        durations = np.asarray(y["time"], dtype=np.float32)
        events = np.asarray(y["event"], dtype=np.int64)
        y_tr = self.labtrans.fit_transform(durations, events)
        self.duration_index = self.labtrans.cuts
        net = tt.practical.MLPVanilla(
            in_features=X.shape[1],
            num_nodes=[64, 32],
            out_features=self.labtrans.out_features,
            batch_norm=False,
            dropout=0.1,
        )
        self.model = DeepHitSingle(
            net,
            tt.optim.Adam(self.lr),
            alpha=0.2,
            sigma=0.1,
            duration_index=self.duration_index,
        )
        self.model.fit(X, y_tr, batch_size=self.batch_size, epochs=self.epochs, verbose=False)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        surv = self.model.predict_surv_df(X)
        risk = 1.0 - surv.iloc[-1].to_numpy(dtype=float)
        return risk

    def predict_survival_function(self, X):
        X = np.asarray(X, dtype=np.float32)
        surv = self.model.predict_surv_df(X)
        times = surv.index.to_numpy(dtype=float)
        vals = surv.to_numpy(dtype=float).T

        class StepFn:
            def __init__(self, t, s):
                self.t = t
                self.s = s

            def __call__(self, x):
                idx = np.searchsorted(self.t, x, side="right") - 1
                if idx < 0:
                    return 1.0
                idx = min(idx, len(self.s) - 1)
                return float(self.s[idx])

        return [StepFn(times, row) for row in vals]


def build_preprocessor():
    numeric_cols = ["Age", "Tumor Size Summary (2016+)"]
    categorical_cols = [c for c in PREDICTOR_COLUMNS if c not in numeric_cols]
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ]
    )


def candidate_models(selected=None):
    # These are the locally available manuscript-aligned algorithms.
    models = {
        "cox_ph": CoxPHSurvivalAnalysis(alpha=1e-4),
        "lasso": CoxnetSurvivalAnalysis(
            l1_ratio=1.0, n_alphas=40, alpha_min_ratio=0.01, max_iter=1000, fit_baseline_model=True
        ),
        "elastic_net": CoxnetSurvivalAnalysis(
            l1_ratio=0.5, n_alphas=40, alpha_min_ratio=0.01, max_iter=1000, fit_baseline_model=True
        ),
        "survival_svm": FastSurvivalSVM(rank_ratio=1.0, alpha=0.01, max_iter=1000, random_state=RANDOM_STATE),
        "gbm": GradientBoostingSurvivalAnalysis(
            learning_rate=0.05,
            n_estimators=200,
            max_depth=2,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "xgboost_survival": XGBoostSurvivalWrapper(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        "lightgbm_survival": LightGBMSurvivalWrapper(
            num_boost_round=300,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
        ),
        "random_survival_forest": RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "superpc": SuperPCSimplified(n_components=3),
        "deephit": DeepHitWrapper(num_durations=32, epochs=8, batch_size=512, lr=1e-2),
    }
    if selected is None:
        return models
    missing = sorted(set(selected) - set(models))
    if missing:
        raise ValueError(f"unknown model(s): {missing}")
    return {k: models[k] for k in selected}


def unavailable_algorithms():
    return {
        "Survival_Quilts_meta_framework": "adapted separately in scripts/train_survival_quilts_official.py; not part of this base-model benchmark runner",
    }


def structured_target(df):
    return Surv.from_arrays(event=df[EVENT_COLUMN].astype(bool).to_numpy(), time=df[TIME_COLUMN].astype(float).to_numpy())


def harrell_cindex(y, risk_scores):
    return float(concordance_index_censored(y["event"], y["time"], risk_scores)[0])


def ipcw_cindex(y_train, y_test, risk_scores):
    return float(concordance_index_ipcw(y_train, y_test, risk_scores)[0])


def predict_risk(model, X):
    # Higher score means higher risk for all models used here.
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def predict_horizon_risk(model, X, horizon):
    if not hasattr(model, "predict_survival_function"):
        return None
    surv_fns = model.predict_survival_function(X)
    risks = []
    for fn in surv_fns:
        surv_prob = float(fn(horizon))
        surv_prob = min(max(surv_prob, 0.0), 1.0)
        risks.append(1.0 - surv_prob)
    return np.asarray(risks, dtype=float)


def horizon_brier(model, X_train, y_train, X_test, y_test, horizon):
    if not hasattr(model, "predict_survival_function"):
        return None
    risks = predict_horizon_risk(model, X_test, horizon)
    surv_probs = 1.0 - risks
    times, scores = brier_score(y_train, y_test, surv_probs, np.asarray([horizon], dtype=float))
    return float(scores[0])


def prepare_dataframe(path):
    df = pd.read_csv(path)
    expected = [ID_COLUMN, TIME_COLUMN, EVENT_COLUMN] + PREDICTOR_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    work = df[expected].copy()
    work = work.dropna(subset=[TIME_COLUMN, EVENT_COLUMN] + PREDICTOR_COLUMNS)
    work = work[(work[TIME_COLUMN] > 0) & (work["Age"].between(18, 90))]

    # Make categorical variables explicit.
    for col in PREDICTOR_COLUMNS:
        if col not in ["Age", "Tumor Size Summary (2016+)"]:
            work[col] = work[col].astype(str)

    work[EVENT_COLUMN] = work[EVENT_COLUMN].astype(int)
    return work


def create_split_manifest(df):
    _, X_temp, idx_train, idx_temp = train_test_split(
        df[PREDICTOR_COLUMNS],
        df.index.to_numpy(),
        test_size=0.36,
        random_state=RANDOM_STATE,
        stratify=df[EVENT_COLUMN],
    )
    _, _, idx_val, idx_test = train_test_split(
        X_temp,
        idx_temp,
        test_size=(20.0 / 36.0),
        random_state=RANDOM_STATE,
        stratify=df.loc[idx_temp, EVENT_COLUMN],
    )
    split_df = df[[ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    split_df["row_index"] = df.index.to_numpy()
    split_df["split"] = "train"
    split_df.loc[idx_val, "split"] = "val"
    split_df.loc[idx_test, "split"] = "test"
    return split_df


def load_split_manifest(df, split_manifest_path):
    manifest = pd.read_csv(split_manifest_path)
    if "row_index" in manifest.columns:
        split_df = manifest.copy()
    else:
        keys = [ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]
        split_df = df[keys].merge(manifest[keys + ["split"]], on=keys, how="left", validate="one_to_one")
        split_df["row_index"] = df.index.to_numpy()
    if split_df["split"].isna().any():
        raise ValueError(f"split manifest has missing assignments: {split_manifest_path}")
    return split_df


def resolve_split_manifest(df, split_manifest_path=None):
    if split_manifest_path:
        split_df = load_split_manifest(df, split_manifest_path)
    else:
        split_df = create_split_manifest(df)
    idx_train = split_df.loc[split_df["split"] == "train", "row_index"].to_numpy()
    idx_val = split_df.loc[split_df["split"] == "val", "row_index"].to_numpy()
    idx_test = split_df.loc[split_df["split"] == "test", "row_index"].to_numpy()
    return split_df, idx_train, idx_val, idx_test


def fit_and_evaluate(df, output_dir, source_path, model_names=None, split_manifest_path=None):
    split_df, idx_train, idx_val, idx_test = resolve_split_manifest(df, split_manifest_path)

    X = df[PREDICTOR_COLUMNS]
    y = structured_target(df)
    X_train = X.loc[idx_train]
    X_val = X.loc[idx_val]
    X_test = X.loc[idx_test]
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    model_map = candidate_models(model_names)
    results = []
    fitted = {}
    train_pred_df = df.loc[idx_train, [ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    val_pred_df = df.loc[idx_val, [ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    test_pred_df = df.loc[idx_test, [ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    for name, estimator in model_map.items():
        model = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", estimator),
            ]
        )
        model.fit(X_train, y_train)
        fitted[name] = model

        train_risk = predict_risk(model, X_train)
        val_risk = predict_risk(model, X_val)
        test_risk = predict_risk(model, X_test)
        train_pred_df[f"risk_{name}"] = train_risk
        val_pred_df[f"risk_{name}"] = val_risk
        test_pred_df[f"risk_{name}"] = test_risk

        row = {
            "model": name,
            "val_harrell_cindex": harrell_cindex(y_val, val_risk),
            "val_ipcw_cindex": ipcw_cindex(y_train, y_val, val_risk),
            "test_harrell_cindex": harrell_cindex(y_test, test_risk),
            "test_ipcw_cindex": ipcw_cindex(y_train, y_test, test_risk),
        }
        for horizon in HORIZONS:
            key = f"test_brier_{int(horizon)}"
            value = horizon_brier(model, X_train, y_train, X_test, y_test, horizon)
            row[key] = value
            train_horizon = predict_horizon_risk(model, X_train, horizon)
            val_horizon = predict_horizon_risk(model, X_val, horizon)
            test_horizon = predict_horizon_risk(model, X_test, horizon)
            if train_horizon is not None:
                train_pred_df[f"risk_{name}_{int(horizon)}"] = train_horizon
            if val_horizon is not None:
                val_pred_df[f"risk_{name}_{int(horizon)}"] = val_horizon
            if test_horizon is not None:
                test_pred_df[f"risk_{name}_{int(horizon)}"] = test_horizon
        results.append(row)

    metrics = pd.DataFrame(results).sort_values(["val_ipcw_cindex", "test_ipcw_cindex"], ascending=False)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]

    score_df = df[[ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    for name, model in fitted.items():
        score_df[f"risk_{name}"] = predict_risk(model, X)

    for horizon in HORIZONS:
        risk = predict_horizon_risk(best_model, X, horizon)
        if risk is not None:
            score_df[f"AIscore{int(horizon)}_retrained"] = risk

    if "AIscore60" in pd.read_csv(source_path, nrows=1).columns and "AIscore60_retrained" in score_df.columns:
        original = pd.read_csv(source_path, usecols=["AIscore60"])["AIscore60"]
        aligned = score_df["AIscore60_retrained"]
        corr = float(pd.concat([original, aligned], axis=1).corr().iloc[0, 1])
    else:
        corr = math.nan

    split_summary = {
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "event_rate_total": float(df[EVENT_COLUMN].mean()),
        "event_rate_train": float(df.loc[idx_train, EVENT_COLUMN].mean()),
        "event_rate_val": float(df.loc[idx_val, EVENT_COLUMN].mean()),
        "event_rate_test": float(df.loc[idx_test, EVENT_COLUMN].mean()),
    }

    summary = {
        "data_file": str(source_path),
        "target_columns": {"time": TIME_COLUMN, "event": EVENT_COLUMN},
        "predictors": PREDICTOR_COLUMNS,
        "split": split_summary,
        "available_algorithms_run": list(model_map.keys()),
        "manuscript_algorithms_unavailable_locally": unavailable_algorithms(),
        "best_model_by_validation_ipcw_cindex": best_name,
        "ai_score60_correlation_with_existing_column": corr,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "seer_model_metrics.csv", index=False)
    score_df.to_csv(output_dir / "seer_retrained_scores.csv", index=False)
    split_df.to_csv(output_dir / "seer_split_manifest.csv", index=False)
    train_pred_df.to_csv(output_dir / "seer_train_predictions.csv", index=False)
    val_pred_df.to_csv(output_dir / "seer_val_predictions.csv", index=False)
    test_pred_df.to_csv(output_dir / "seer_test_predictions.csv", index=False)
    with open(output_dir / "seer_training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    joblib.dump(
        {
            "models": fitted,
            "predictors": PREDICTOR_COLUMNS,
            "time_column": TIME_COLUMN,
            "event_column": EVENT_COLUMN,
            "horizons": HORIZONS,
            "random_state": RANDOM_STATE,
        },
        output_dir / "python_models.joblib",
    )
    joblib.dump(
        {
            "model": best_model,
            "best_model_name": best_name,
            "predictors": PREDICTOR_COLUMNS,
            "time_column": TIME_COLUMN,
            "event_column": EVENT_COLUMN,
            "horizons": HORIZONS,
            "random_state": RANDOM_STATE,
        },
        output_dir / "best_model.joblib",
    )
    return metrics, summary


def main():
    parser = argparse.ArgumentParser(description="Train AI-clinical score models on SEER.csv.")
    parser.add_argument("--input", default="AIdata/SEER.csv")
    parser.add_argument("--output-dir", default="outputs/ai_clinical_score")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to train. Default runs all available Python-side models.",
    )
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Optional CSV manifest with row_index or Patient ID/time/status/split columns.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    df = prepare_dataframe(input_path)
    selected = [x.strip() for x in args.models.split(",") if x.strip()] or None
    split_manifest = args.split_manifest or None
    metrics, summary = fit_and_evaluate(df, output_dir, input_path, selected, split_manifest)

    print("Training finished.")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
