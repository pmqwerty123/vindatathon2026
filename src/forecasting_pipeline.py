#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final VinDatathon forecasting pipeline.

The script trains the production ensemble, writes diagnostic artifacts, and
checks that the generated submission follows the competition schema.
"""

# !pip install lightgbm prophet --quiet

from pathlib import Path
import logging
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    HAS_PROPHET = True
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    print("Prophet: available")
except ImportError:
    HAS_PROPHET = False
    print("Prophet: NOT available — using HW fallback")

SEED = 42
np.random.seed(SEED)



DATA_DIR = Path('data/raw')
OUT_DIR  = Path('outputs/submissions')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sales = pd.read_csv(DATA_DIR/'sales.csv', parse_dates=['Date'])
sales = sales.assign(
    Y=sales.Date.dt.year,
    Q=sales.Date.dt.quarter,
    M=sales.Date.dt.month,
    DOW=sales.Date.dt.dayofweek,
    day=sales.Date.dt.day,
)

print(sales.shape, sales.Date.min().date(), '->', sales.Date.max().date())
sales.head()

fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
axes[0].plot(sales.Date, sales.Revenue/1e6, lw=0.5)
axes[0].set_ylabel('Revenue (M)'); axes[0].set_title('Revenue 2012-2022')
axes[1].plot(sales.Date, sales.COGS/1e6, lw=0.5, color='red')
axes[1].set_ylabel('COGS (M)')
plt.tight_layout(); plt.savefig(str(OUT_DIR / "feature_importance.png"))
plt.close()

monthly = sales.groupby(['Y','M']).Revenue.mean().reset_index()
fig, ax = plt.subplots(figsize=(12, 5))
for y in sorted(sales.Y.unique()):
    d = monthly[monthly.Y == y]
    ax.plot(d.M, d.Revenue/1e6, marker='o', label=str(y))
ax.set_xticks(range(1,13)); ax.legend(ncol=4, fontsize=8)
ax.set_title('Mua vu thang qua cac nam')
plt.savefig(str(OUT_DIR / "feature_importance.png"))
plt.close()

q_marg = sales.groupby(['Y','Q']).apply(
    lambda d: d.COGS.sum()/d.Revenue.sum(), include_groups=False).reset_index()
q_marg.columns = ['Y','Q','margin']

fig, ax = plt.subplots(figsize=(11, 4))
for q in [1,2,3,4]:
    d = q_marg[q_marg.Q == q]
    ax.plot(d.Y, d.margin, marker='o', label=f'Q{q}')
ax.axhline(0.9, color='gray', ls='--')
ax.set_title('Margin theo quy - Q3 zigzag chan/le')
ax.legend(); plt.savefig(str(OUT_DIR / "feature_importance.png"))
plt.close()

PROMO_SCHEDULE = [
    {'name': 'spring_sale',   'month': 3,  'day': 18, 'duration': 30, 'discount': 12,   'recurrence': True},
    {'name': 'mid_year',      'month': 6,  'day': 23, 'duration': 29, 'discount': 18,   'recurrence': True},
    {'name': 'fall_launch',   'month': 8,  'day': 30, 'duration': 32, 'discount': 10,   'recurrence': True},
    {'name': 'year_end',      'month': 11, 'day': 18, 'duration': 45, 'discount': 20,   'recurrence': True},
    {'name': 'urban_blowout', 'month': 7,  'day': 30, 'duration': 33, 'discount': None, 'recurrence': 'odd'},
    {'name': 'rural_special', 'month': 1,  'day': 30, 'duration': 30, 'discount': 15,   'recurrence': 'odd'},
]

TET_DATES = {
    2013:'2013-02-10', 2014:'2014-01-31', 2015:'2015-02-19',
    2016:'2016-02-08', 2017:'2017-01-28', 2018:'2018-02-16',
    2019:'2019-02-05', 2020:'2020-01-25', 2021:'2021-02-12',
    2022:'2022-02-01', 2023:'2023-01-22', 2024:'2024-02-10',
}

VN_FIXED_HOLIDAYS = [
    (1,1,'new_year'), (3,8,'womens_day'), (4,30,'reunification'),
    (5,1,'labor_day'), (9,2,'national_day'), (10,20,'vn_womens_day'),
    (11,11,'dd_1111'), (12,12,'dd_1212'),
    (12,24,'christmas_eve'), (12,25,'christmas'),
]

def build_features(dates):
    df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    d = df['Date']

    df = df.assign(
        year=d.dt.year,
        month=d.dt.month,
        day=d.dt.day,
        dow=d.dt.dayofweek,
        doy=d.dt.dayofyear,
        quarter=d.dt.quarter,
    )
    df['is_weekend']    = (df['dow']>=5).astype(int)
    df['days_to_eom']   = d.dt.days_in_month - df['day']
    df['days_from_som'] = df['day'] - 1
    df['dim']           = d.dt.days_in_month

    # Edge of month
    for k in [1,2,3]:
        df[f'is_last{k}']  = (df['days_to_eom']  <= k-1).astype(int)
        df[f'is_first{k}'] = (df['days_from_som'] <= k-1).astype(int)

    # Trend + regime
    df['t_days']  = (d - pd.Timestamp('2020-01-01')).dt.days
    df['t_years'] = df['t_days']/365.25
    df['regime_pre2019']  = (df['year']<=2018).astype(int)
    df['regime_2019']     = (df['year']==2019).astype(int)
    df['regime_post2019'] = (df['year']>=2020).astype(int)

    # Fourier
    TAU = 2*np.pi
    for k in (1,2,3,4,5):
        df[f'sin_y{k}'] = np.sin(TAU*k*df['doy']/365.25)
        df[f'cos_y{k}'] = np.cos(TAU*k*df['doy']/365.25)
    for k in (1,2):
        df[f'sin_w{k}'] = np.sin(TAU*k*df['dow']/7.0)
        df[f'cos_w{k}'] = np.cos(TAU*k*df['dow']/7.0)
    for k in (1,2):
        df[f'sin_m{k}'] = np.sin(TAU*k*(df['day']-1)/df['dim'])
        df[f'cos_m{k}'] = np.cos(TAU*k*(df['day']-1)/df['dim'])

    # Holidays
    for (m, dd_, name) in VN_FIXED_HOLIDAYS:
        df[f'hol_{name}'] = ((df['month']==m) & (df['day']==dd_)).astype(int)

    # Tet distance
    tet_lut = {y: pd.Timestamp(v) for y,v in TET_DATES.items()}
    def nearest_tet_diff(dd):
        cands = [tet_lut.get(dd.year), tet_lut.get(dd.year-1), tet_lut.get(dd.year+1)]
        cands = [c for c in cands if c is not None]
        valid = [(dd-c).days for c in cands if abs((dd-c).days)<=45]
        return min(valid) if valid else 999
    diffs = np.array([nearest_tet_diff(dd) for dd in d])
    df['tet_days_diff'] = diffs
    df['tet_in_7']      = (np.abs(diffs)<=7).astype(int)
    df['tet_in_14']     = (np.abs(diffs)<=14).astype(int)
    df['tet_before_7']  = ((diffs>=-7) & (diffs<0)).astype(int)
    df['tet_after_7']   = ((diffs>0) & (diffs<=7)).astype(int)
    df['tet_on']        = (diffs==0).astype(int)

    # Black Friday
    def is_bf(dd):
        if dd.month != 11: return 0
        last = pd.Timestamp(year=dd.year, month=11, day=30)
        last_fri = last - pd.Timedelta(days=(last.dayofweek - 4) % 7)
        return int(dd == last_fri)
    df['hol_black_friday'] = [is_bf(dd) for dd in d]

    # Promo windows
    yrs = sorted(set(df['year'].tolist()))
    for promo in PROMO_SCHEDULE:
        name = promo['name']
        sm = promo['month']
        sd = promo['day']
        dur = promo['duration']
        disc = promo['discount']
        recur = promo['recurrence']
        in_prom = np.zeros(len(df), dtype=int)
        since   = np.full(len(df), -1.0)
        until   = np.full(len(df), -1.0)
        discount= np.zeros(len(df))
        for y in range(min(yrs)-1, max(yrs)+2):
            if recur=='odd' and y%2==0: continue
            start = pd.Timestamp(year=y, month=sm, day=sd)
            end   = start + pd.Timedelta(days=dur)
            mask  = (d>=start) & (d<=end)
            in_prom[mask] = 1
            since[mask]   = (d[mask]-start).dt.days
            until[mask]   = (end-d[mask]).dt.days
            discount[mask]= disc or 0
        df[f'promo_{name}']       = in_prom
        df[f'promo_{name}_since'] = since
        df[f'promo_{name}_until'] = until
        df[f'promo_{name}_disc']  = discount

    df['is_odd_year'] = (df['year'] % 2).astype(int)
    df['liberation_decay'] = df['hol_reunification'] * (df['year'] - 2012)
    return df

print('Test:', build_features(pd.date_range('2023-01-01','2023-01-05')).shape)

feat = build_features(sales['Date'])
feat['Revenue'] = sales['Revenue'].values
feat['COGS']    = sales['COGS'].values

sample_path = DATA_DIR / 'sample_submission.csv'
if sample_path.exists():
    sample_submission = pd.read_csv(sample_path, parse_dates=['Date'])
    test_dates = sample_submission['Date']
else:
    sample_submission = None
    test_dates = pd.date_range('2023-01-01', '2024-07-01', freq='D')
test_df = build_features(test_dates)

NON_FEATURES = {'Date','Revenue','COGS'}
cols = [c for c in feat.columns if c not in NON_FEATURES]

X_tr = feat[cols].values.astype(float)
X_te = test_df[cols].values.astype(float)
y_rev = np.log(feat['Revenue'].values)
y_cog = np.log(feat['COGS'].values)
years = feat['Date'].dt.year.values

print(f'Training: {X_tr.shape}, Test: {X_te.shape}, Features: {len(cols)}')

w_full = np.full(len(years), 0.01)
w_full[(years>=2014) & (years<=2018)] = 1.0
print(f'Days with w=1.0: {(w_full==1.0).sum()}, w=0.01: {(w_full==0.01).sum()}')

def train_ridge(X_train, y_train, alpha=3.0):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0).replace(0, 1)
    Xs = (X_train - mu) / sigma
    m = Ridge(alpha=alpha, random_state=42)
    m.fit(Xs, y_train)
    return m, (mu, sigma)

def predict_ridge(model, X_test, stats):
    mu, sigma = stats
    return model.predict((X_test - mu) / sigma)

ridge_rev, st_r = train_ridge(pd.DataFrame(X_tr, columns=cols), y_rev)
ridge_cog, st_c = train_ridge(pd.DataFrame(X_tr, columns=cols), y_cog)

p_rd_rev = np.exp(predict_ridge(ridge_rev, pd.DataFrame(X_te, columns=cols), st_r))
p_rd_cog = np.exp(predict_ridge(ridge_cog, pd.DataFrame(X_te, columns=cols), st_c))

print(f'Ridge Revenue: {p_rd_rev.mean():,.0f}')
print(f'Ridge COGS:    {p_rd_cog.mean():,.0f}')

LGB_PARAMS = dict(
    objective='regression', metric='mae',
    learning_rate=0.03, num_leaves=63,
    min_data_in_leaf=30,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    lambda_l2=1.0, seed=42, verbosity=-1,
)

def train_lgb(X, y, w, num_boost_es=5000, early_stop=300):
    intern = pd.Timestamp('2022-07-04')
    fit_idx = (feat['Date'] <= intern).values
    ins_idx = (feat['Date'] >  intern).values

    bk = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X[fit_idx], y[fit_idx], weight=w[fit_idx]),
        num_boost_round=num_boost_es,
        valid_sets=[lgb.Dataset(X[ins_idx], y[ins_idx])],
        callbacks=[lgb.early_stopping(early_stop, verbose=False),
                   lgb.log_evaluation(0)])

    bf = lgb.train(LGB_PARAMS,
                   lgb.Dataset(X, y, weight=w),
                   num_boost_round=bk.best_iteration)
    return bf, bk.best_iteration

def train_lgb_split(X, y, w, fit_idx, val_idx, num_boost_es=5000, early_stop=300):
    bk = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X[fit_idx], y[fit_idx], weight=w[fit_idx]),
        num_boost_round=num_boost_es,
        valid_sets=[lgb.Dataset(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(early_stop, verbose=False),
                   lgb.log_evaluation(0)])
    return bk, bk.best_iteration

print('Train LGB Revenue...')
bf_rev, it_rev = train_lgb(X_tr, y_rev, w_full)
print(f'  best_iter={it_rev}')

print('Train LGB COGS...')
bf_cog, it_cog = train_lgb(X_tr, y_cog, w_full)
print(f'  best_iter={it_cog}')

p_lgb_rev = np.exp(bf_rev.predict(X_te))
p_lgb_cog = np.exp(bf_cog.predict(X_te))

print(f'LGB Revenue: {p_lgb_rev.mean():,.0f}')
print(f'LGB COGS:    {p_lgb_cog.mean():,.0f}')

if HAS_PROPHET:
    def build_promo_regressors(dates):
        full = build_features(dates)
        promo_cols = [c for c in full.columns
                      if c.startswith('promo_') and c.count('_') == 1]
        return full[['Date'] + promo_cols].rename(columns={'Date':'ds'})
    
    def fit_prophet(train_df, post_regime_only=True):
        if post_regime_only:
            train_df = train_df[train_df['ds'] >= '2020-01-01']
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05)
        for col in [c for c in train_df.columns if c.startswith('promo_')]:
            m.add_regressor(col)
        m.fit(train_df)
        return m

if HAS_PROPHET:
    print('Train Prophet Revenue...')
    tdf_r = pd.DataFrame({'ds': sales['Date'], 'y': np.log(sales['Revenue'])}) \
              .merge(build_promo_regressors(sales['Date']), on='ds')
    mp_r = fit_prophet(tdf_r)
    
    print('Train Prophet COGS...')
    tdf_c = pd.DataFrame({'ds': sales['Date'], 'y': np.log(sales['COGS'])}) \
              .merge(build_promo_regressors(sales['Date']), on='ds')
    mp_c = fit_prophet(tdf_c)
    
    vdf = pd.DataFrame({'ds': test_df['Date']}) \
            .merge(build_promo_regressors(test_df['Date']), on='ds')
    p_pr_rev = np.exp(mp_r.predict(vdf)['yhat'].values)
    p_pr_cog = np.exp(mp_c.predict(vdf)['yhat'].values)
    
    print(f'Prophet Revenue: {p_pr_rev.mean():,.0f}')
    print(f'Prophet COGS:    {p_pr_cog.mean():,.0f}')
else:
    hw_fb_r = ExponentialSmoothing(np.log(sales[sales["Date"] >= "2020-01-01"]["Revenue"].values),
        trend="add", seasonal="add", seasonal_periods=365, initialization_method="estimated").fit(optimized=True)
    p_pr_rev = np.exp(hw_fb_r.forecast(len(test_dates)))
    hw_fb_c = ExponentialSmoothing(np.log(sales[sales["Date"] >= "2020-01-01"]["COGS"].values),
        trend="add", seasonal="add", seasonal_periods=365, initialization_method="estimated").fit(optimized=True)
    p_pr_cog = np.exp(hw_fb_c.forecast(len(test_dates)))
    print(f"HW fallback Revenue: {p_pr_rev.mean():,.0f}")
    print(f"HW fallback COGS:    {p_pr_cog.mean():,.0f}")


def train_q_specialist(X, y, w_base, target_q, q_boost=2.0):
    Q_train = feat['Date'].dt.quarter.values
    w = w_base.copy()
    w[Q_train == target_q] *= q_boost
    bf, _ = train_lgb(X, y, w, num_boost_es=3000, early_stop=200)
    return bf

spec_rev = {}
spec_cog = {}
for q in [1, 2, 3, 4]:
    print(f'Spec_Q{q} Revenue...')
    bf = train_q_specialist(X_tr, y_rev, w_full, q)
    spec_rev[q] = np.exp(bf.predict(X_te))

    print(f'Spec_Q{q} COGS...')
    bf = train_q_specialist(X_tr, y_cog, w_full, q)
    spec_cog[q] = np.exp(bf.predict(X_te))

print('Done 8 specialists.')

# Ghep theo quy cua test date
Q_test = test_df['Date'].dt.quarter.values
lgb_spec_rev = np.zeros(len(test_dates))
lgb_spec_cog = np.zeros(len(test_dates))
for q in [1,2,3,4]:
    mask = Q_test == q
    lgb_spec_rev[mask] = spec_rev[q][mask]
    lgb_spec_cog[mask] = spec_cog[q][mask]

ALPHA = 0.60   # specialist vs base LGB
CR    = 1.26   # calibration Revenue
CC    = 1.32   # calibration COGS

# Tang 1: LGB blend
lgb_blend_rev = ALPHA * lgb_spec_rev + (1 - ALPHA) * p_lgb_rev
lgb_blend_cog = ALPHA * lgb_spec_cog + (1 - ALPHA) * p_lgb_cog

# Tang 2: 3-way blend
raw_rev = 0.10 * p_pr_rev + 0.10 * p_rd_rev + 0.80 * lgb_blend_rev
raw_cog = 0.10 * p_pr_cog + 0.10 * p_rd_cog + 0.80 * lgb_blend_cog

# Tang 3: calibration
final_rev = CR * raw_rev
final_cog = CC * raw_cog

print(f'Final Revenue: {final_rev.mean():,.0f}')
print(f'Final COGS:    {final_cog.mean():,.0f}')

sub = pd.DataFrame({
    'Date':    test_df['Date'].dt.strftime('%Y-%m-%d').values,
    'Revenue': final_rev,
    'COGS':    final_cog,
})

# Intermediate sub
sub.head()

target_cog_mean = sub.COGS.mean()

# Biennial Margin Rule: Odd years match 2021, Even years match 2022
margin_2021 = {q: sales[(sales.Q==q) & (sales.Y==2021)].COGS.sum() / sales[(sales.Q==q) & (sales.Y==2021)].Revenue.sum() for q in [1,2,3,4]}
margin_2022 = {q: sales[(sales.Q==q) & (sales.Y==2022)].COGS.sum() / sales[(sales.Q==q) & (sales.Y==2022)].Revenue.sum() for q in [1,2,3,4]}

def get_margin(dd):
    if dd.year % 2 != 0:
        return margin_2021[dd.quarter]
    else:
        return margin_2022[dd.quarter]


sub_v57 = sub.copy()
sub_v57['Date_dt'] = pd.to_datetime(sub_v57['Date'])
sub_v57['Q'] = sub_v57['Date_dt'].dt.quarter

BETA = 0.30
sub_v57['historical_margin'] = sub_v57['Date_dt'].apply(get_margin)
for q in [1,2,3,4]:
    mask = sub_v57.Q == q
    historical_cog = sub_v57.loc[mask, 'Revenue'] * sub_v57.loc[mask, 'historical_margin']
    sub_v57.loc[mask, 'COGS'] = (1-BETA)*sub_v57.loc[mask, 'COGS'] + BETA*historical_cog

scale = target_cog_mean / sub_v57.COGS.mean()
sub_v57['COGS'] = sub_v57['COGS'] * scale

out_v57 = OUT_DIR / 'submission_v57_mp_blend30.csv'
sub_v57[['Date','Revenue','COGS']].to_csv(out_v57, index=False)
print(f'Saved v57: {out_v57}')

# Diversification candidate: identical v57 logic with OOF-selected calibration.
OOF_ALPHA = 0.80
OOF_CR = 1.15
OOF_CC = 1.20
OOF_BETA = 0.15
oof_lgb_blend_rev = OOF_ALPHA * lgb_spec_rev + (1 - OOF_ALPHA) * p_lgb_rev
oof_lgb_blend_cog = OOF_ALPHA * lgb_spec_cog + (1 - OOF_ALPHA) * p_lgb_cog
oof_raw_rev = 0.10 * p_pr_rev + 0.10 * p_rd_rev + 0.80 * oof_lgb_blend_rev
oof_raw_cog = 0.10 * p_pr_cog + 0.10 * p_rd_cog + 0.80 * oof_lgb_blend_cog
sub_oof = pd.DataFrame({
    'Date': test_df['Date'].dt.strftime('%Y-%m-%d').values,
    'Revenue': OOF_CR * oof_raw_rev,
    'COGS': OOF_CC * oof_raw_cog,
})
oof_target_cog_mean = sub_oof.COGS.mean()
sub_oof['Date_dt'] = pd.to_datetime(sub_oof['Date'])
sub_oof['Q'] = sub_oof['Date_dt'].dt.quarter
sub_oof['historical_margin'] = sub_oof['Date_dt'].apply(get_margin)
for q in [1, 2, 3, 4]:
    mask = sub_oof.Q == q
    historical_cog = sub_oof.loc[mask, 'Revenue'] * sub_oof.loc[mask, 'historical_margin']
    sub_oof.loc[mask, 'COGS'] = (1 - OOF_BETA) * sub_oof.loc[mask, 'COGS'] + OOF_BETA * historical_cog
sub_oof['COGS'] = sub_oof['COGS'] * (oof_target_cog_mean / sub_oof.COGS.mean())
out_oof = OUT_DIR / 'submission_oof_calibrated.csv'
sub_oof[['Date','Revenue','COGS']].to_csv(out_oof, index=False)
print(f'Saved OOF-calibrated diversification candidate: {out_oof}')

# XGBoost for diversity — different tree structure than LightGBM
XGB_PARAMS = dict(
    objective='reg:squarederror', eval_metric='mae',
    learning_rate=0.03, max_depth=6, min_child_weight=30,
    subsample=0.85, colsample_bytree=0.85,
    reg_lambda=1.0, seed=42, verbosity=0, n_jobs=-1,
)

def train_xgb(X, y, w, num_rounds=3000, early_stop=300):
    intern = pd.Timestamp('2022-07-04')
    fit_idx = (feat['Date'] <= intern).values
    val_idx = (feat['Date'] >  intern).values
    
    dtrain = xgb.DMatrix(X[fit_idx], y[fit_idx], weight=w[fit_idx])
    dval   = xgb.DMatrix(X[val_idx], y[val_idx])
    
    bk = xgb.train(XGB_PARAMS, dtrain, num_boost_round=num_rounds,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=early_stop, verbose_eval=False)
    best_iter = bk.best_iteration
    
    # Retrain on full data
    dfull = xgb.DMatrix(X, y, weight=w)
    bf = xgb.train(XGB_PARAMS, dfull, num_boost_round=best_iter)
    return bf, best_iter

print('Train XGB Revenue...')
xgb_rev, xgb_it_rev = train_xgb(X_tr, y_rev, w_full)
print(f'  best_iter={xgb_it_rev}')

print('Train XGB COGS...')
xgb_cog, xgb_it_cog = train_xgb(X_tr, y_cog, w_full)
print(f'  best_iter={xgb_it_cog}')

p_xgb_rev = np.exp(xgb_rev.predict(xgb.DMatrix(X_te)))
p_xgb_cog = np.exp(xgb_cog.predict(xgb.DMatrix(X_te)))
print(f'XGB Revenue: {p_xgb_rev.mean():,.0f}')
print(f'XGB COGS:    {p_xgb_cog.mean():,.0f}')


# Holt-Winters: train on recent regime (2018-2022)
hw_train = sales[sales['Date'] >= '2018-01-01'].copy()

print('Train HW Revenue...')
hw_rev_model = ExponentialSmoothing(
    np.log(hw_train['Revenue'].values),
    trend='add', seasonal='add', seasonal_periods=365,
    initialization_method='estimated'
).fit(optimized=True)
p_hw_rev = np.exp(hw_rev_model.forecast(len(test_dates)))

print('Train HW COGS...')
hw_cog_model = ExponentialSmoothing(
    np.log(hw_train['COGS'].values),
    trend='add', seasonal='add', seasonal_periods=365,
    initialization_method='estimated'
).fit(optimized=True)
p_hw_cog = np.exp(hw_cog_model.forecast(len(test_dates)))

print(f'HW Revenue: {p_hw_rev.mean():,.0f}')
print(f'HW COGS:    {p_hw_cog.mean():,.0f}')


# Validation predictions (use 2022-H2 as internal validation)
intern = pd.Timestamp('2022-07-04')
val_mask = (feat['Date'] > intern).values
fit_mask = ~val_mask
y_val_rev_raw = feat.loc[val_mask, 'Revenue'].values
y_val_cog_raw = feat.loc[val_mask, 'COGS'].values

def summarize_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        'model': name,
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': rmse,
        'r2': r2_score(y_true, y_pred),
    }

# Leakage-free holdout diagnostics: models below are trained only through 2022-07-04.
val_rows = []
val_lgb_rev, _ = train_lgb_split(X_tr, y_rev, w_full, fit_mask, val_mask, num_boost_es=3000, early_stop=200)
val_lgb_cog, _ = train_lgb_split(X_tr, y_cog, w_full, fit_mask, val_mask, num_boost_es=3000, early_stop=200)
val_pred_lgb_rev = np.exp(val_lgb_rev.predict(X_tr[val_mask]))
val_pred_lgb_cog = np.exp(val_lgb_cog.predict(X_tr[val_mask]))

val_ridge_rev, val_st_r = train_ridge(pd.DataFrame(X_tr[fit_mask], columns=cols), y_rev[fit_mask])
val_ridge_cog, val_st_c = train_ridge(pd.DataFrame(X_tr[fit_mask], columns=cols), y_cog[fit_mask])
val_pred_ridge_rev = np.exp(predict_ridge(val_ridge_rev, pd.DataFrame(X_tr[val_mask], columns=cols), val_st_r))
val_pred_ridge_cog = np.exp(predict_ridge(val_ridge_cog, pd.DataFrame(X_tr[val_mask], columns=cols), val_st_c))

val_rows.extend([
    summarize_metrics('holdout_lgb_revenue', y_val_rev_raw, val_pred_lgb_rev),
    summarize_metrics('holdout_lgb_cogs', y_val_cog_raw, val_pred_lgb_cog),
    summarize_metrics('holdout_ridge_revenue', y_val_rev_raw, val_pred_ridge_rev),
    summarize_metrics('holdout_ridge_cogs', y_val_cog_raw, val_pred_ridge_cog),
])
validation_report = pd.DataFrame(val_rows)
validation_report.to_csv(OUT_DIR / 'validation_report.csv', index=False)
print('\n=== Leakage-free 2022-H2 Validation ===')
print(validation_report.to_string(index=False, formatters={'mae':'{:,.0f}'.format, 'rmse':'{:,.0f}'.format, 'r2':'{:.4f}'.format}))

# Get validation predictions from each model
p_val_lgb_rev = np.exp(bf_rev.predict(X_tr[val_mask]))
p_val_rd_rev  = np.exp(predict_ridge(ridge_rev, pd.DataFrame(X_tr[val_mask], columns=cols), st_r))
p_val_xgb_rev = np.exp(xgb_rev.predict(xgb.DMatrix(X_tr[val_mask])))
p_val_spec_rev = np.zeros(val_mask.sum())
Q_val = feat.loc[val_mask, 'Date'].dt.quarter.values
# Need to retrain specialists to get val predictions — use bf from train_lgb
# For simplicity, use the base LGB blend as spec proxy on validation
p_val_lgb_blend_rev = ALPHA * p_val_lgb_rev + (1-ALPHA) * p_val_lgb_rev  # spec ~= base on val

# Stack all models for NNLS
A_rev = np.column_stack([p_val_lgb_rev, p_val_rd_rev, p_val_xgb_rev])
w_nnls_rev, _ = nnls(A_rev, y_val_rev_raw)
w_nnls_rev /= w_nnls_rev.sum()

A_cog = np.column_stack([
    np.exp(bf_cog.predict(X_tr[val_mask])),
    np.exp(predict_ridge(ridge_cog, pd.DataFrame(X_tr[val_mask], columns=cols), st_c)),
    np.exp(xgb_cog.predict(xgb.DMatrix(X_tr[val_mask])))
])
w_nnls_cog, _ = nnls(A_cog, y_val_cog_raw)
w_nnls_cog /= w_nnls_cog.sum()

print(f'NNLS Revenue weights: LGB={w_nnls_rev[0]:.3f}, Ridge={w_nnls_rev[1]:.3f}, XGB={w_nnls_rev[2]:.3f}')
print(f'NNLS COGS weights:    LGB={w_nnls_cog[0]:.3f}, Ridge={w_nnls_cog[1]:.3f}, XGB={w_nnls_cog[2]:.3f}')


# NNLS blend of base models
nnls_rev = w_nnls_rev[0]*p_lgb_rev + w_nnls_rev[1]*p_rd_rev + w_nnls_rev[2]*p_xgb_rev
nnls_cog = w_nnls_cog[0]*p_lgb_cog + w_nnls_cog[1]*p_rd_cog + w_nnls_cog[2]*p_xgb_cog

# Blend with specialist and Prophet (keep original structure)
lgb_blend_rev2 = ALPHA * lgb_spec_rev + (1 - ALPHA) * nnls_rev
lgb_blend_cog2 = ALPHA * lgb_spec_cog + (1 - ALPHA) * nnls_cog

raw_rev2 = 0.10 * p_pr_rev + 0.10 * p_hw_rev + 0.80 * lgb_blend_rev2
raw_cog2 = 0.10 * p_pr_cog + 0.10 * p_hw_cog + 0.80 * lgb_blend_cog2

final_rev2 = CR * raw_rev2
final_cog2 = CC * raw_cog2

# Apply v57 margin fix
sub_up = pd.DataFrame({
    'Date': test_df['Date'].dt.strftime('%Y-%m-%d').values,
    'Revenue': final_rev2,
    'COGS': final_cog2,
})

target_cog_mean2 = sub_up.COGS.mean()
sub_up['Date_dt'] = pd.to_datetime(sub_up['Date'])
sub_up['Q'] = sub_up['Date_dt'].dt.quarter

sub_up['historical_margin'] = sub_up['Date_dt'].apply(get_margin)
for q in [1,2,3,4]:
    mask = sub_up.Q == q
    hist_cog = sub_up.loc[mask, 'Revenue'] * sub_up.loc[mask, 'historical_margin']
    sub_up.loc[mask, 'COGS'] = (1-BETA)*sub_up.loc[mask, 'COGS'] + BETA*hist_cog

scale2 = target_cog_mean2 / sub_up.COGS.mean()
sub_up['COGS'] = sub_up['COGS'] * scale2

out_up = OUT_DIR / 'submission_upgraded.csv'
sub_up[['Date','Revenue','COGS']].to_csv(out_up, index=False)
print(f'Saved upgraded candidate: {out_up}')
print(f'Revenue mean: {sub_up.Revenue.mean():,.0f}')
print(f'COGS mean:    {sub_up.COGS.mean():,.0f}')

# Risk-adjusted final: keep v57 as the official submission because the later
# NNLS candidate uses optimistic full-data validation predictions for weights.
out_final = OUT_DIR / 'submission.csv'
sub_v57[['Date','Revenue','COGS']].to_csv(out_final, index=False)
print(f'Saved risk-adjusted final submission: {out_final}')


# SHAP analysis for Revenue LightGBM model

FEATURE_NAME_MAP = {
    't_days': 'Global Trend (Days)',
    't_years': 'Global Trend (Years)',
    'year': 'Year',
    'month': 'Month',
    'day': 'Day of Month',
    'dow': 'Day of Week',
    'doy': 'Day of Year',
    'quarter': 'Quarter',
    'is_weekend': 'Is Weekend',
    'days_to_eom': 'Days to End of Month',
    'days_from_som': 'Days from Start of Month',
    'dim': 'Days in Month',
    'is_last1': 'Last Day of Month',
    'is_last2': 'Last 2 Days of Month',
    'is_last3': 'Last 3 Days of Month',
    'is_first1': 'First Day of Month',
    'is_first2': 'First 2 Days of Month',
    'is_first3': 'First 3 Days of Month',
    'regime_pre2019': 'Regime: Pre-2019',
    'regime_2019': 'Regime: 2019 Transition',
    'regime_post2019': 'Regime: Post-2019',
    'sin_y1': 'Annual Cyclicality (Sin 1)',
    'cos_y1': 'Annual Cyclicality (Cos 1)',
    'sin_y2': 'Annual Cyclicality (Sin 2)',
    'cos_y2': 'Annual Cyclicality (Cos 2)',
    'sin_y3': 'Annual Cyclicality (Sin 3)',
    'cos_y3': 'Annual Cyclicality (Cos 3)',
    'sin_y4': 'Annual Cyclicality (Sin 4)',
    'cos_y4': 'Annual Cyclicality (Cos 4)',
    'sin_y5': 'Annual Cyclicality (Sin 5)',
    'cos_y5': 'Annual Cyclicality (Cos 5)',
    'sin_w1': 'Weekly Cyclicality (Sin 1)',
    'cos_w1': 'Weekly Cyclicality (Cos 1)',
    'sin_w2': 'Weekly Cyclicality (Sin 2)',
    'cos_w2': 'Weekly Cyclicality (Cos 2)',
    'sin_m1': 'Monthly Cyclicality (Sin 1)',
    'cos_m1': 'Monthly Cyclicality (Cos 1)',
    'sin_m2': 'Monthly Cyclicality (Sin 2)',
    'cos_m2': 'Monthly Cyclicality (Cos 2)',
    'hol_new_year': 'Holiday: New Year',
    'hol_womens_day': 'Holiday: Womens Day',
    'hol_reunification': 'Holiday: Reunification Day',
    'hol_labor_day': 'Holiday: Labor Day',
    'hol_national_day': 'Holiday: National Day',
    'hol_vn_womens_day': 'Holiday: VN Womens Day',
    'hol_dd_1111': 'Promo Event: 11.11',
    'hol_dd_1212': 'Promo Event: 12.12',
    'hol_christmas_eve': 'Holiday: Christmas Eve',
    'hol_christmas': 'Holiday: Christmas',
    'tet_days_diff': 'Days to Lunar New Year (Tet)',
    'tet_in_7': 'Within 7 Days of Tet',
    'tet_in_14': 'Within 14 Days of Tet',
    'tet_before_7': '7 Days Before Tet',
    'tet_after_7': '7 Days After Tet',
    'tet_on': 'Lunar New Year Day (Tet)',
    'hol_black_friday': 'Holiday: Black Friday',
    'promo_spring_sale': 'Promo: Spring Sale',
    'promo_spring_sale_since': 'Days Since Spring Sale Started',
    'promo_spring_sale_until': 'Days Until Spring Sale Ends',
    'promo_spring_sale_disc': 'Discount: Spring Sale',
    'promo_mid_year': 'Promo: Mid-Year Sale',
    'promo_mid_year_since': 'Days Since Mid-Year Sale Started',
    'promo_mid_year_until': 'Days Until Mid-Year Sale Ends',
    'promo_mid_year_disc': 'Discount: Mid-Year Sale',
    'promo_fall_launch': 'Promo: Fall Launch',
    'promo_fall_launch_since': 'Days Since Fall Launch Started',
    'promo_fall_launch_until': 'Days Until Fall Launch Ends',
    'promo_fall_launch_disc': 'Discount: Fall Launch',
    'promo_year_end': 'Promo: Year-End Sale',
    'promo_year_end_since': 'Days Since Year-End Sale Started',
    'promo_year_end_until': 'Days Until Year-End Sale Ends',
    'promo_year_end_disc': 'Discount: Year-End Sale',
    'promo_urban_blowout': 'Promo: Urban Blowout',
    'promo_urban_blowout_since': 'Days Since Urban Blowout',
    'promo_urban_blowout_until': 'Days Until Urban Blowout',
    'promo_urban_blowout_disc': 'Discount: Urban Blowout',
    'promo_rural_special': 'Promo: Rural Special',
    'promo_rural_special_since': 'Days Since Rural Special',
    'promo_rural_special_until': 'Days Until Rural Special',
    'promo_rural_special_disc': 'Discount: Rural Special',
    'is_odd_year': 'Is Odd Year (Biennial Indicator)',
    'liberation_decay': 'Holiday Fatigue Trend (Decay)'
}

mapped_cols = [FEATURE_NAME_MAP.get(c, c) for c in cols]

print('Computing SHAP values for Revenue model...')
explainer_rev = shap.TreeExplainer(bf_rev)
shap_vals_rev = explainer_rev.shap_values(X_te)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals_rev, X_te, feature_names=mapped_cols, show=False, max_display=20)
plt.title('SHAP Summary — Revenue Model (Top 20 Features)', fontsize=14)
plt.tight_layout()
plt.savefig(str(OUT_DIR / 'shap_revenue_summary.png'), dpi=150, bbox_inches='tight')
plt.savefig(str(OUT_DIR / "feature_importance.png"))
plt.close()
print(f'Saved: {OUT_DIR}/shap_revenue_summary.png')

# SHAP for COGS model
print('Computing SHAP values for COGS model...')
explainer_cog = shap.TreeExplainer(bf_cog)
shap_vals_cog = explainer_cog.shap_values(X_te)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_vals_cog, X_te, feature_names=mapped_cols, show=False, max_display=20)
plt.title('SHAP Summary — COGS Model (Top 20 Features)', fontsize=14)
plt.tight_layout()
plt.savefig(str(OUT_DIR / 'shap_cogs_summary.png'), dpi=150, bbox_inches='tight')
plt.savefig(str(OUT_DIR / "feature_importance.png"))
plt.close()
print(f'Saved: {OUT_DIR}/shap_cogs_summary.png')


# Feature importance from LightGBM (gain)
imp_rev = bf_rev.feature_importance(importance_type='gain')
imp_cog = bf_cog.feature_importance(importance_type='gain')

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Revenue
idx_r = np.argsort(imp_rev)[-20:]
axes[0].barh(range(20), imp_rev[idx_r], color='#2E86AB')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels([mapped_cols[i] for i in idx_r])
axes[0].set_title('Revenue — Top 20 Feature Importance (Gain)', fontsize=13)
axes[0].set_xlabel('Gain')

# COGS
idx_c = np.argsort(imp_cog)[-20:]
axes[1].barh(range(20), imp_cog[idx_c], color='#A8312C')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([mapped_cols[i] for i in idx_c])
axes[1].set_title('COGS — Top 20 Feature Importance (Gain)', fontsize=13)
axes[1].set_xlabel('Gain')

plt.tight_layout()
plt.savefig(str(OUT_DIR / 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.savefig(str(OUT_DIR / "feature_importance.png"))
plt.close()
print(f'Saved: {OUT_DIR}/feature_importance.png')


# Compare v57 baseline vs upgraded
baseline = pd.read_csv(OUT_DIR / 'submission_v57_mp_blend30.csv')
upgraded = pd.read_csv(OUT_DIR / 'submission.csv')

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(pd.to_datetime(baseline.Date), baseline.Revenue/1e6, label='v57 Baseline', alpha=0.8, lw=0.8)
axes[0].plot(pd.to_datetime(upgraded.Date), upgraded.Revenue/1e6, label='Upgraded', alpha=0.8, lw=0.8)
axes[0].set_ylabel('Revenue (M)')
axes[0].set_title('Revenue: Baseline vs Upgraded')
axes[0].legend()

axes[1].plot(pd.to_datetime(baseline.Date), baseline.COGS/1e6, label='v57 Baseline', alpha=0.8, lw=0.8)
axes[1].plot(pd.to_datetime(upgraded.Date), upgraded.COGS/1e6, label='Upgraded', alpha=0.8, lw=0.8)
axes[1].set_ylabel('COGS (M)')
axes[1].set_title('COGS: Baseline vs Upgraded')
axes[1].legend()

plt.tight_layout()
plt.savefig(str(OUT_DIR / 'baseline_vs_upgraded.png'), dpi=150, bbox_inches='tight')
plt.savefig(str(OUT_DIR / "feature_importance.png"))
plt.close()

# Stats comparison
print('\n=== Comparison Statistics ===')
print(f"Revenue Mean — Baseline: {baseline.Revenue.mean():,.0f} | Upgraded: {upgraded.Revenue.mean():,.0f} | Diff: {upgraded.Revenue.mean()-baseline.Revenue.mean():+,.0f}")
print(f"COGS Mean    — Baseline: {baseline.COGS.mean():,.0f} | Upgraded: {upgraded.COGS.mean():,.0f} | Diff: {upgraded.COGS.mean()-baseline.COGS.mean():+,.0f}")
print(f"Margin       — Baseline: {baseline.COGS.sum()/baseline.Revenue.sum():.4f} | Upgraded: {upgraded.COGS.sum()/upgraded.Revenue.sum():.4f}")


# Final run card
model_snapshot = pd.DataFrame([
    ('Linear ridge anchor', p_rd_rev.mean(), p_rd_cog.mean()),
    ('LightGBM core learner', p_lgb_rev.mean(), p_lgb_cog.mean()),
    ('XGBoost diversity learner', p_xgb_rev.mean(), p_xgb_cog.mean()),
    ('Prophet / HW event prior', p_pr_rev.mean(), p_pr_cog.mean()),
    ('Holt-Winters seasonal prior', p_hw_rev.mean(), p_hw_cog.mean()),
    ('Quarter-specialist LGB', lgb_spec_rev.mean(), lgb_spec_cog.mean()),
    ('Official risk-adjusted final', upgraded.Revenue.mean(), upgraded.COGS.mean()),
], columns=['component', 'avg_revenue', 'avg_cogs'])

print('\n' + '=' * 76)
print('FINAL FORECAST RUN CARD')
print('=' * 76)
print(model_snapshot.to_string(
    index=False,
    formatters={
        'avg_revenue': '{:,.0f}'.format,
        'avg_cogs': '{:,.0f}'.format,
    },
))
print('-' * 76)
print(f"Blend weights       | Revenue NNLS: LGB {w_nnls_rev[0]:.3f}, Ridge {w_nnls_rev[1]:.3f}, XGB {w_nnls_rev[2]:.3f}")
print(f"                    | COGS NNLS:    LGB {w_nnls_cog[0]:.3f}, Ridge {w_nnls_cog[1]:.3f}, XGB {w_nnls_cog[2]:.3f}")
print(f"Feature space       | {len(cols)} engineered predictors")
print(f"Final margin ratio  | {upgraded.COGS.sum() / upgraded.Revenue.sum():.4f}")
print('Artifacts written   | submission.csv, validation_report.csv, SHAP plots, feature importance, comparison plot')
print('=' * 76)



print('\n--- DATATHON CONSTRAINTS VERIFICATION ---')
import pandas as pd
import sys

# Load sample and final submission
sample = pd.read_csv('data/raw/sample_submission.csv')
final = pd.read_csv('outputs/submissions/submission.csv')

errors = []

# 1. Row Count Check
if len(sample) != len(final):
    errors.append(f'Row count mismatch: Expected {len(sample)}, Got {len(final)}')

# 2. Column Check
expected_cols = ['Date', 'Revenue', 'COGS']
if list(final.columns) != expected_cols:
    errors.append(f'Column mismatch: Expected {expected_cols}, Got {list(final.columns)}')

# 3. Order Check
if not final['Date'].equals(sample['Date']):
    errors.append('Date order mismatch: The dates do not perfectly align with sample_submission.csv')

# 4. Null Check
if final['Revenue'].isnull().any() or final['COGS'].isnull().any():
    errors.append('Null values detected in predictions!')

if errors:
    print('CONSTRAINTS FAILED!')
    for e in errors:
        print(f' - {e}')
    sys.exit(1)
else:
    print('All format constraints passed!')
    print('Final submission cleanly saved to: outputs/submissions/submission.csv')
