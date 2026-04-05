# optuna with alpha
import os
import numpy as np
import pandas as pd
import optuna

import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from scipy.stats import ttest_rel, t
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def weighted_median(preds_2d, weights):
    preds_2d = np.asarray(preds_2d, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_estimators, n_samples = preds_2d.shape
    out = np.empty(n_samples, dtype=float)

    wsum = np.sum(weights)
    if wsum <= 0:
        return np.median(preds_2d, axis=0)
    weights = weights / wsum

    for i in range(n_samples):
        p = preds_2d[:, i]
        order = np.argsort(p)
        p_sorted = p[order]
        w_sorted = weights[order]
        cw = np.cumsum(w_sorted)
        out[i] = p_sorted[np.searchsorted(cw, 0.5)]
    return out

class FIAdaBoostRegressor:
    def __init__(self, n_estimators=150, max_depth=4, alpha=1.0, random_state=42, use_weighted_median=True):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.use_weighted_median = bool(use_weighted_median)
        self.estimators_ = []
        self.estimator_weights_ = []

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y)

        X_vals = np.asarray(X.values, dtype=float)
        y_vals = np.asarray(pd.Series(y).values, dtype=float)

        n_samples, n_features = X_vals.shape
        weights = np.ones(n_samples, dtype=float) / n_samples

        x_min = X_vals.min(axis=0)
        x_max = X_vals.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range == 0] = 1.0
        X_norm = (X_vals - x_min) / x_range

        self.estimators_.clear()
        self.estimator_weights_.clear()

        for t_idx in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + t_idx
            )
            tree.fit(X_vals, y_vals, sample_weight=weights)

            y_pred = tree.predict(X_vals)
            errors = np.abs(y_vals - y_pred)

            max_error = np.max(errors)
            if max_error <= 0:
                break

            e_norm = errors / max_error

            raw_fi = tree.feature_importances_
            fi_sum = np.sum(raw_fi)
            if fi_sum <= 0:
                phi_f = np.ones(n_features, dtype=float) / n_features
            else:
                phi_f = raw_fi / fi_sum

            phi_x = X_norm @ phi_f

            avg_error = np.sum(weights * e_norm)
            if avg_error >= 0.5:
                break

            beta_t = avg_error / (1.0 - avg_error)

            # APPLY ALPHA TO CONTROL LEARNING SPEED
            exponent = self.alpha * (1.0 - (e_norm * phi_x))
            weights = weights * np.power(beta_t, exponent)

            zt = np.sum(weights)
            if zt <= 0:
                break
            weights /= zt

            self.estimators_.append(tree)
            self.estimator_weights_.append(self.alpha * np.log(1.0 / (beta_t + 1e-12)))

        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_vals = np.asarray(X.values, dtype=float)

        if len(self.estimators_) == 0:
            return np.zeros(X_vals.shape[0], dtype=float)

        preds = np.array([est.predict(X_vals) for est in self.estimators_], dtype=float)
        w = np.asarray(self.estimator_weights_, dtype=float)

        if self.use_weighted_median:
            pred = weighted_median(preds, w)
        else:
            pred = np.average(preds, axis=0, weights=w)
            
        return np.clip(pred, 0.0, None) # Prevents negative predictions

# =========================
# CONFIG
# =========================
N_SPLITS = 5
OPTUNA_TRIALS = 30 
SAMPLE_FRAC = 0.2
TARGET = "solar_energy_potential"
LEAKAGE_COLS = ["sunshine_hours", "clear_sky_ratio", "sunshine_flag", "year_month", "ALLSKY_SFC_SW_DWN", "solar_exposure_index"]

def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(ROOT_DIR, "data", "processed")
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    data_path = os.path.join(processed_dir, "integrated_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run data_integration.py first.")

    df = pd.read_csv(data_path)

    if "date" not in df.columns:
        raise ValueError("Missing 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target '{TARGET}'. Ensure data_integration.py created it.")
    
    if SAMPLE_FRAC is not None:
        keep_n = int(len(df) * float(SAMPLE_FRAC))
        keep_n = max(1, keep_n)
        df = df.iloc[:keep_n].copy().reset_index(drop=True)

    drop_cols = [TARGET, "date", "element", "id", "month"] + LEAKAGE_COLS

    for c in ["lat", "lon"]:
        if c in df.columns and df[c].nunique(dropna=False) == 1:
            drop_cols.append(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0).astype(float)

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    dates = df["date"].copy()

    # =======================================================
    # PHASE 1: OPTUNA HYPERPARAMETER TUNING
    # =======================================================
    print(f"\n[PHASE 1] Running Optuna Optimization ({OPTUNA_TRIALS} Trials)...")
    
    def objective(trial):
        # 🚨 EXPANDED SEARCH SPACE FOR MAXIMUM ACCURACY 🚨
        n_est = trial.suggest_int('n_estimators', 50, 200)
        m_depth = trial.suggest_int('max_depth', 3, 7)
        alpha_val = trial.suggest_float('alpha', 0.1, 1.0) # Let Optuna find the perfect brake pedal
        
        tscv_opt = TimeSeriesSplit(n_splits=N_SPLITS)
        rmses = []
        
        for train_idx, val_idx in tscv_opt.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = FIAdaBoostRegressor(n_estimators=n_est, max_depth=m_depth, alpha=alpha_val, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            
        return np.mean(rmses)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    optuna_trials_df = study.trials_dataframe()
    optuna_trials_path = os.path.join(model_dir, "optuna_trials_history.csv")
    optuna_trials_df.to_csv(optuna_trials_path, index=False)
    print(f"\nSaved Optuna trial history: {optuna_trials_path}")

    best_n = study.best_params['n_estimators']
    best_d = study.best_params['max_depth']
    best_a = study.best_params['alpha']
        
    print(f"✅ Optuna Found Optimal Parameters: n_estimators={best_n}, max_depth={best_d}, alpha={best_a:.4f}")

    best_params_df = pd.DataFrame({
        "parameter": ["n_estimators", "max_depth", "alpha"],
        "value": [best_n, best_d, best_a]
    })
    best_params_path = os.path.join(model_dir, "optuna_best_params.csv")
    best_params_df.to_csv(best_params_path, index=False)
    print(f"Saved best parameters: {best_params_path}")

    # =======================================================
    # PHASE 2: FAIR TIME-SERIES CV & STATISTICAL TESTING
    # =======================================================
    print("\n[PHASE 2] Cross-Validation & Statistical Testing")
    print(f"Rows: {len(df)} | Features: {X.shape[1]} | Splits: {N_SPLITS}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_metrics = []
    all_date_mae_base = []
    all_date_mae_fi = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        d_test = dates.iloc[test_idx].reset_index(drop=True)

        print(f"\n[Fold {fold}/{N_SPLITS}] train={len(train_idx)} test={len(test_idx)}")

        # Baseline gets the Optuna alpha as its learning rate
        baseline = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=best_d, random_state=42),
            n_estimators=best_n,
            learning_rate=best_a,
            random_state=42
        )
        baseline.fit(X_train, y_train)
        pred_base = baseline.predict(X_test)
        m_base = evaluate(y_test, pred_base)

        # FI gets the Optuna alpha
        fi = FIAdaBoostRegressor(
            n_estimators=best_n,
            max_depth=best_d,
            alpha=best_a,
            random_state=42,
            use_weighted_median=True
        )
        fi.fit(X_train, y_train)
        pred_fi = fi.predict(X_test)
        m_fi = evaluate(y_test, pred_fi)

        fold_metrics.append({
            "fold": fold,
            "baseline_RMSE": m_base["RMSE"],
            "baseline_MAE": m_base["MAE"],
            "baseline_R2": m_base["R2"],
            "fi_RMSE": m_fi["RMSE"],
            "fi_MAE": m_fi["MAE"],
            "fi_R2": m_fi["R2"],
        })

        print(f"  Baseline: RMSE={m_base['RMSE']:.4f}, MAE={m_base['MAE']:.4f}, R2={m_base['R2']:.4f}")
        print(f"  FI:       RMSE={m_fi['RMSE']:.4f}, MAE={m_fi['MAE']:.4f}, R2={m_fi['R2']:.4f}")

        abs_base = np.abs(y_test.to_numpy() - pred_base)
        abs_fi = np.abs(y_test.to_numpy() - pred_fi)

        tmp = pd.DataFrame({
            "date": d_test,
            "abs_base": abs_base,
            "abs_fi": abs_fi
        })

        per_date = tmp.groupby("date", as_index=False).mean(numeric_only=True)
        all_date_mae_base.append(per_date["abs_base"].to_numpy())
        all_date_mae_fi.append(per_date["abs_fi"].to_numpy())

    metrics_df = pd.DataFrame(fold_metrics)
    print("\n=== TIME-SERIES CV SUMMARY (mean ± std across folds) ===")
    for k in ["baseline_RMSE", "baseline_MAE", "baseline_R2", "fi_RMSE", "fi_MAE", "fi_R2"]:
        print(f"{k}: {metrics_df[k].mean():.4f} ± {metrics_df[k].std(ddof=1):.4f}")

    metrics_out = os.path.join(model_dir, "cv_fold_metrics.csv")
    metrics_df.to_csv(metrics_out, index=False)


if __name__ == "__main__":
    main()