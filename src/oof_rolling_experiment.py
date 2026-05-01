#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controlled rolling-origin experiment for DATATHON 2026.

This script does not promote any model to submission.csv. It stress-tests the
v57-style formula against fold-local calibration choices and writes diagnostics
to outputs/submissions/oof_rolling_experiment_results.csv.
"""

from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path("data/raw")
OUT_DIR = Path("outputs/submissions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMO_SCHEDULE = [
    ("spring_sale", 3, 18, 30, 12, True),
    ("mid_year", 6, 23, 29, 18, True),
    ("fall_launch", 8, 30, 32, 10, True),
    ("year_end", 11, 18, 45, 20, True),
    ("urban_blowout", 7, 30, 33, None, "odd"),
    ("rural_special", 1, 30, 30, 15, "odd"),
]

TET_DATES = {
    2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
}

VN_FIXED_HOLIDAYS = [
    (1, 1, "new_year"), (3, 8, "womens_day"), (4, 30, "reunification"),
    (5, 1, "labor_day"), (9, 2, "national_day"), (10, 20, "vn_womens_day"),
    (11, 11, "dd_1111"), (12, 12, "dd_1212"),
    (12, 24, "christmas_eve"), (12, 25, "christmas"),
]

LGB_PARAMS = dict(
    objective="regression", metric="mae", learning_rate=0.03, num_leaves=63,
    min_data_in_leaf=30, feature_fraction=0.85, bagging_fraction=0.85,
    bagging_freq=5, lambda_l2=1.0, seed=SEED, verbosity=-1,
)


def build_features(dates):
    df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    d = df["Date"]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["days_to_eom"] = d.dt.days_in_month - df["day"]
    df["days_from_som"] = df["day"] - 1
    df["dim"] = d.dt.days_in_month
    for k in [1, 2, 3]:
        df[f"is_last{k}"] = (df["days_to_eom"] <= k - 1).astype(int)
        df[f"is_first{k}"] = (df["days_from_som"] <= k - 1).astype(int)
    df["t_days"] = (d - pd.Timestamp("2020-01-01")).dt.days
    df["t_years"] = df["t_days"] / 365.25
    df["regime_pre2019"] = (df["year"] <= 2018).astype(int)
    df["regime_2019"] = (df["year"] == 2019).astype(int)
    df["regime_post2019"] = (df["year"] >= 2020).astype(int)
    tau = 2 * np.pi
    for k in (1, 2, 3, 4, 5):
        df[f"sin_y{k}"] = np.sin(tau * k * df["doy"] / 365.25)
        df[f"cos_y{k}"] = np.cos(tau * k * df["doy"] / 365.25)
    for k in (1, 2):
        df[f"sin_w{k}"] = np.sin(tau * k * df["dow"] / 7.0)
        df[f"cos_w{k}"] = np.cos(tau * k * df["dow"] / 7.0)
        df[f"sin_m{k}"] = np.sin(tau * k * (df["day"] - 1) / df["dim"])
        df[f"cos_m{k}"] = np.cos(tau * k * (df["day"] - 1) / df["dim"])
    for m, dd, name in VN_FIXED_HOLIDAYS:
        df[f"hol_{name}"] = ((df["month"] == m) & (df["day"] == dd)).astype(int)
    tet_lut = {y: pd.Timestamp(v) for y, v in TET_DATES.items()}
    def nearest_tet_diff(dd):
        cands = [tet_lut.get(dd.year), tet_lut.get(dd.year - 1), tet_lut.get(dd.year + 1)]
        valid = [(dd - c).days for c in cands if c is not None and abs((dd - c).days) <= 45]
        return min(valid) if valid else 999
    diffs = np.array([nearest_tet_diff(dd) for dd in d])
    df["tet_days_diff"] = diffs
    df["tet_in_7"] = (np.abs(diffs) <= 7).astype(int)
    df["tet_in_14"] = (np.abs(diffs) <= 14).astype(int)
    df["tet_before_7"] = ((diffs >= -7) & (diffs < 0)).astype(int)
    df["tet_after_7"] = ((diffs > 0) & (diffs <= 7)).astype(int)
    df["tet_on"] = (diffs == 0).astype(int)
    def is_bf(dd):
        if dd.month != 11:
            return 0
        last = pd.Timestamp(year=dd.year, month=11, day=30)
        last_fri = last - pd.Timedelta(days=(last.dayofweek - 4) % 7)
        return int(dd == last_fri)
    df["hol_black_friday"] = [is_bf(dd) for dd in d]
    yrs = sorted(set(df["year"].tolist()))
    for name, sm, sd, dur, disc, recur in PROMO_SCHEDULE:
        in_prom = np.zeros(len(df), dtype=int)
        since = np.full(len(df), -1.0)
        until = np.full(len(df), -1.0)
        discount = np.zeros(len(df))
        for y in range(min(yrs) - 1, max(yrs) + 2):
            if recur == "odd" and y % 2 == 0:
                continue
            start = pd.Timestamp(year=y, month=sm, day=sd)
            end = start + pd.Timedelta(days=dur)
            mask = (d >= start) & (d <= end)
            in_prom[mask] = 1
            since[mask] = (d[mask] - start).dt.days
            until[mask] = (end - d[mask]).dt.days
            discount[mask] = disc or 0
        df[f"promo_{name}"] = in_prom
        df[f"promo_{name}_since"] = since
        df[f"promo_{name}_until"] = until
        df[f"promo_{name}_disc"] = discount
    df["is_odd_year"] = (df["year"] % 2).astype(int)
    df["liberation_decay"] = df["hol_reunification"] * (df["year"] - 2012)
    return df


def metric_dict(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def train_lgb_fold(X, y, w, train_idx, val_idx, rounds=3000, early_stop=200):
    model = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X[train_idx], y[train_idx], weight=w[train_idx]),
        num_boost_round=rounds,
        valid_sets=[lgb.Dataset(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_ridge_fold(X, y, train_idx):
    Xdf = pd.DataFrame(X)
    mu = Xdf.iloc[train_idx].mean(axis=0)
    sigma = Xdf.iloc[train_idx].std(axis=0).replace(0, 1)
    model = Ridge(alpha=3.0, random_state=SEED)
    model.fit(((Xdf.iloc[train_idx] - mu) / sigma).values, y[train_idx])
    return model, mu, sigma


def pred_ridge_fold(model, mu, sigma, X):
    return model.predict(((pd.DataFrame(X) - mu) / sigma).values)


def hw_forecast(train_values, horizon):
    # Use HW only when enough history exists for a full annual season.
    try:
        if len(train_values) < 730:
            raise ValueError("not enough history")
        model = ExponentialSmoothing(
            np.log(train_values), trend="add", seasonal="add", seasonal_periods=365,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.exp(model.forecast(horizon))
    except Exception:
        return np.repeat(train_values[-365:].mean(), horizon)


def margin_fix(df_pred, train_sales, beta):
    out = df_pred.copy()
    target_mean = out["COGS"].mean()
    out["Date_dt"] = pd.to_datetime(out["Date"])
    out["Q"] = out["Date_dt"].dt.quarter
    available_years = sorted(train_sales["Y"].unique())
    odd_years = [y for y in available_years if y % 2 == 1]
    even_years = [y for y in available_years if y % 2 == 0]
    odd_ref = max(odd_years) if odd_years else max(available_years)
    even_ref = max(even_years) if even_years else max(available_years)
    margins = {}
    for ref_name, ref_year in [("odd", odd_ref), ("even", even_ref)]:
        margins[ref_name] = {}
        for q in [1, 2, 3, 4]:
            d = train_sales[(train_sales["Y"] == ref_year) & (train_sales["Q"] == q)]
            margins[ref_name][q] = d["COGS"].sum() / d["Revenue"].sum()
    hist_margin = []
    for dd in out["Date_dt"]:
        hist_margin.append(margins["odd" if dd.year % 2 else "even"][dd.quarter])
    hist_cogs = out["Revenue"].values * np.array(hist_margin)
    out["COGS"] = (1 - beta) * out["COGS"].values + beta * hist_cogs
    out["COGS"] *= target_mean / out["COGS"].mean()
    return out[["Date", "Revenue", "COGS"]]


def build_fold_predictions(sales, feat, cols, fold_year):
    train_idx = (feat["Date"].dt.year < fold_year).values
    val_idx = (feat["Date"].dt.year == fold_year).values
    X = feat[cols].values.astype(float)
    y_rev = np.log(feat["Revenue"].values)
    y_cog = np.log(feat["COGS"].values)
    years = feat["Date"].dt.year.values
    w = np.full(len(years), 0.01)
    w[(years >= 2014) & (years <= 2018)] = 1.0
    lgb_rev = train_lgb_fold(X, y_rev, w, train_idx, val_idx)
    lgb_cog = train_lgb_fold(X, y_cog, w, train_idx, val_idx)
    p_lgb_rev = np.exp(lgb_rev.predict(X[val_idx]))
    p_lgb_cog = np.exp(lgb_cog.predict(X[val_idx]))
    spec_rev = np.zeros(val_idx.sum())
    spec_cog = np.zeros(val_idx.sum())
    q_val = feat.loc[val_idx, "Date"].dt.quarter.values
    q_train = feat["Date"].dt.quarter.values
    for q in [1, 2, 3, 4]:
        wq = w.copy()
        wq[q_train == q] *= 2.0
        mr = train_lgb_fold(X, y_rev, wq, train_idx, val_idx, rounds=2000, early_stop=150)
        mc = train_lgb_fold(X, y_cog, wq, train_idx, val_idx, rounds=2000, early_stop=150)
        mask = q_val == q
        spec_rev[mask] = np.exp(mr.predict(X[val_idx][mask]))
        spec_cog[mask] = np.exp(mc.predict(X[val_idx][mask]))
    rr, mu_r, sig_r = train_ridge_fold(X, y_rev, train_idx)
    rc, mu_c, sig_c = train_ridge_fold(X, y_cog, train_idx)
    p_rd_rev = np.exp(pred_ridge_fold(rr, mu_r, sig_r, X[val_idx]))
    p_rd_cog = np.exp(pred_ridge_fold(rc, mu_c, sig_c, X[val_idx]))
    train_sales = sales.loc[train_idx].copy()
    p_hw_rev = hw_forecast(train_sales["Revenue"].values, val_idx.sum())
    p_hw_cog = hw_forecast(train_sales["COGS"].values, val_idx.sum())
    return {
        "fold_year": fold_year,
        "train_sales": train_sales,
        "dates": feat.loc[val_idx, "Date"].dt.strftime("%Y-%m-%d").values,
        "y_rev_true": feat.loc[val_idx, "Revenue"].values,
        "y_cog_true": feat.loc[val_idx, "COGS"].values,
        "p_lgb_rev": p_lgb_rev,
        "p_lgb_cog": p_lgb_cog,
        "spec_rev": spec_rev,
        "spec_cog": spec_cog,
        "p_rd_rev": p_rd_rev,
        "p_rd_cog": p_rd_cog,
        "p_hw_rev": p_hw_rev,
        "p_hw_cog": p_hw_cog,
    }


def score_fold_cache(cache, alpha, cr, cc, beta):
    lgb_blend_rev = alpha * cache["spec_rev"] + (1 - alpha) * cache["p_lgb_rev"]
    lgb_blend_cog = alpha * cache["spec_cog"] + (1 - alpha) * cache["p_lgb_cog"]
    raw_rev = 0.10 * cache["p_hw_rev"] + 0.10 * cache["p_rd_rev"] + 0.80 * lgb_blend_rev
    raw_cog = 0.10 * cache["p_hw_cog"] + 0.10 * cache["p_rd_cog"] + 0.80 * lgb_blend_cog
    pred = pd.DataFrame({
        "Date": cache["dates"],
        "Revenue": cr * raw_rev,
        "COGS": cc * raw_cog,
    })
    pred = margin_fix(pred, cache["train_sales"], beta)
    mr = metric_dict(cache["y_rev_true"], pred["Revenue"].values)
    mc = metric_dict(cache["y_cog_true"], pred["COGS"].values)
    combined_mae = 0.5 * (mr["mae"] + mc["mae"])
    return combined_mae, mr, mc, pred


def main():
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    sales["Y"] = sales["Date"].dt.year
    sales["Q"] = sales["Date"].dt.quarter
    feat = build_features(sales["Date"])
    feat["Revenue"] = sales["Revenue"].values
    feat["COGS"] = sales["COGS"].values
    cols = [c for c in feat.columns if c not in {"Date", "Revenue", "COGS"}]

    folds = [2019, 2020, 2021, 2022]
    default_params = dict(alpha=0.60, cr=1.26, cc=1.32, beta=0.30)
    grid = []
    for alpha in [0.40, 0.60, 0.80]:
        for cr in [1.15, 1.26, 1.35]:
            for cc in [1.20, 1.32, 1.44]:
                for beta in [0.15, 0.30, 0.45]:
                    grid.append(dict(alpha=alpha, cr=cr, cc=cc, beta=beta))

    rows = []
    for fold in folds:
        print(f"\n=== Fold {fold} ===")
        cache = build_fold_predictions(sales, feat, cols, fold)
        base_mae, base_r, base_c, _ = score_fold_cache(cache, **default_params)
        rows.append({"fold": fold, "candidate": "v57_default", "combined_mae": base_mae,
                     "rev_mae": base_r["mae"], "cogs_mae": base_c["mae"], **default_params})
        print(f"v57_default combined_mae={base_mae:,.0f} rev_mae={base_r['mae']:,.0f} cogs_mae={base_c['mae']:,.0f}")
        best = (base_mae, default_params, base_r, base_c)
        # Keep the experiment controlled: search calibration only; model predictions are fold-local.
        for params in grid:
            mae, mr, mc, _ = score_fold_cache(cache, **params)
            if mae < best[0]:
                best = (mae, params, mr, mc)
        rows.append({"fold": fold, "candidate": "best_grid", "combined_mae": best[0],
                     "rev_mae": best[2]["mae"], "cogs_mae": best[3]["mae"], **best[1]})
        print(f"best_grid combined_mae={best[0]:,.0f} improvement={base_mae-best[0]:,.0f} params={best[1]}")

    result = pd.DataFrame(rows)
    out = OUT_DIR / "oof_rolling_experiment_results.csv"
    result.to_csv(out, index=False)
    pivot = result.pivot(index="fold", columns="candidate", values="combined_mae")
    pivot["improvement"] = pivot["v57_default"] - pivot["best_grid"]
    print("\n=== Summary ===")
    print(pivot.to_string(float_format=lambda x: f"{x:,.0f}"))
    print(f"mean_improvement={pivot['improvement'].mean():,.0f}")
    print(f"std_improvement={pivot['improvement'].std(ddof=1):,.0f}")
    print(f"positive_folds={(pivot['improvement'] > 0).sum()}/{len(pivot)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()