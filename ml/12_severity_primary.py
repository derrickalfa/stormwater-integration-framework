# -*- coding: utf-8 -*-
"""
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
from config import Paths, Params

Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
================================================================================
MANHOLE-BASED SPLIT REGRESSION — 8 MODELS
No interaction features | Silt_Depth excluded
================================================================================
✓ LightGBM        (skip if already in checkpoint)
✓ XGBoost         (skip if already in checkpoint)
✓ CatBoost        (skip if already in checkpoint)
✓ Random Forest   (skip if already in checkpoint)
✓ Gradient Boosting (skip if already in checkpoint)
✓ MLP Neural Network
✓ AdaBoost
✓ SVR             (subsampled for speed)
================================================================================
FIXES:
  - load_checkpoint / save_checkpoint / print_result moved BEFORE LOAD DATA
  - make_json_serializable added to handle DecisionTreeRegressor in params
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings, os, pickle, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("✓ Optuna")
except ImportError:
    raise SystemExit("❌ pip install optuna")

try:
    import lightgbm as lgb;  print("✓ LightGBM")
except ImportError:
    lgb = None

try:
    import xgboost as xgb;  print("✓ XGBoost")
except ImportError:
    xgb = None

try:
    import catboost as cb;  print("✓ CatBoost")
except ImportError:
    cb = None


print(f"CPU cores available: {os.cpu_count()}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE    = str(Paths.FINAL_REGRESSION)
OUTPUT_DIR   = str(Paths.ML_SEVERITY_DIR)

RANDOM_STATE = 42
SYSTEM       = 'Stormwater'
TARGET       = 'Surcharge_Ratio'
VAL_FRAC     = 0.15
TEST_FRAC    = 0.20

N_FAST   = 100
N_MEDIUM = 30
N_SLOW   = 15
N_SVR    = 15

SVR_TRIAL_N = 4_000
SVR_FINAL_N = 10_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'training_checkpoint.json')
SHAP_ONLY = False   # Set True to skip training and run SHAP only

# ==============================================================================
# FEATURE LIST — FIXED SORTED ORDER  (never use set() on feature lists)
# ==============================================================================

MAINTENANCE_FEATURES = sorted([
    'Blockages_SinceLastInsp', 'General_Cleaning_SinceLastInsp',
    'Total_Maintenance_Events_SinceLastInsp', 'Inspection_SinceLastInsp'
])
PIPE_FEATURES = sorted([
    'Downstream_Pipe_Age_Avg', 'Downstream_Pipe_Diameter_Avg', 
    'Upstream_Pipe_Diameter_Avg',
    'Downstream_Pipe_Length_Avg', 'Upstream_Pipe_Length_Avg',
    'Downstream_Pipe_Slope_Avg', 'Upstream_Pipe_Slope_Avg'
])
TOPOGRAPHIC_FEATURES    = sorted(['Elev',])
LANDUSE_FEATURES        = sorted(['PopDensity_m2', 'AADT'])
MANHOLE_CONFIG_FEATURES = sorted([
    'Num_Inlet_Pipes', 'Reservoir_Distance'
])
WATER_SILT_CLIMATE_FEATURES = sorted([
    'Silt_Coverage_Ratio', 
    # Silt_Depth intentionally excluded
    'Temp_Mean_7d_Avg',
    'Rain_7d_Sum', 'Windspeed_Day', 'Dewpoint_Day'
])
CATEGORICAL_FEATURES = sorted(['Geol_Type'])

_seen = set(); ALL_FEATURES = []
for f in (MAINTENANCE_FEATURES + PIPE_FEATURES + TOPOGRAPHIC_FEATURES +
          LANDUSE_FEATURES + MANHOLE_CONFIG_FEATURES +
          WATER_SILT_CLIMATE_FEATURES + CATEGORICAL_FEATURES):
    if f not in _seen:
        ALL_FEATURES.append(f); _seen.add(f)

print(f"\n📋 Features: {len(ALL_FEATURES)}  (fixed order, no Silt_Depth, no interactions)")

# ==============================================================================
# HELPERS  — ALL defined here, before LOAD DATA
# ==============================================================================

def safe_label_encode(tr, te):
    ts = tr.astype(str).str.strip(); vs = te.astype(str).str.strip()
    le = LabelEncoder().fit(ts)
    unseen = ~vs.isin(le.classes_)
    if unseen.any():
        vs = vs.copy(); vs[unseen] = ts.value_counts().index[0]
    return le.transform(ts), le.transform(vs), le

def encode_cats_numeric(X_tr, X_va, X_te, cat_cols):
    Xtr, Xva, Xte = X_tr.copy(), X_va.copy(), X_te.copy()
    encoders = {}
    for col in cat_cols:
        tr_e, va_e, le = safe_label_encode(Xtr[col], Xva[col])
        Xtr[col] = tr_e; Xva[col] = va_e
        ts = Xte[col].astype(str).str.strip()
        ts[~ts.isin(le.classes_)] = le.classes_[0]
        Xte[col] = le.transform(ts)
        encoders[col] = le
    return Xtr, Xva, Xte, encoders

def scale_numeric(X_tr, X_va, X_te, cat_cols):
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    Xtr, Xva, Xte = X_tr.copy(), X_va.copy(), X_te.copy()
    sc = StandardScaler()
    Xtr[num_cols] = sc.fit_transform(Xtr[num_cols])
    Xva[num_cols] = sc.transform(Xva[num_cols])
    Xte[num_cols] = sc.transform(Xte[num_cols])
    return Xtr, Xva, Xte, sc

def subsample_rows(X, y, n, seed=RANDOM_STATE):
    if n is None or len(X) <= n:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

def make_json_serializable(obj):
    """Recursively convert params/metrics to JSON-safe types.
    Sklearn estimator objects (e.g. DecisionTreeRegressor) become repr strings."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)  # catches DecisionTreeRegressor, etc.

def save_results(name, model, params, metrics, history):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUTPUT_DIR, f'{name.replace(" ","_")}_{ts}.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'model_name': name,
                     'best_params': params, 'metrics': metrics, 'timestamp': ts}, f)
    with open(os.path.join(OUTPUT_DIR, f'{name.replace(" ","_")}_params.json'), 'w') as f:
        json.dump({'model_name': name,
                   'best_params': make_json_serializable(params),
                   'metrics':     make_json_serializable(metrics),
                   'timestamp': ts}, f, indent=2)
    if history:
        pd.DataFrame(history).to_csv(
            os.path.join(OUTPUT_DIR, f'{name.replace(" ","_")}_trials.csv'), index=False)
    sf = os.path.join(OUTPUT_DIR, 'optimization_summary.csv')
    row = pd.DataFrame([{'Model': name, 'R²': metrics['R²'], 'MAE': metrics['MAE'],
                         'RMSE': metrics['RMSE'],
                         'Training_Time_s': metrics.get('training_time', 0),
                         'N_Trials': len(history) if history else 0,
                         'Timestamp': ts}])
    (pd.concat([pd.read_csv(sf), row], ignore_index=True) if os.path.exists(sf) else row
     ).to_csv(sf, index=False)
    print(f"  ✓ Saved: {name}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f: return json.load(f)
    return {'completed_models': []}

def save_checkpoint(done):
    with open(CHECKPOINT_FILE, 'w') as f: json.dump({'completed_models': done}, f)

def print_result(name, val_r2, test_r2):
    gap = val_r2 - test_r2
    flag = '✅' if abs(gap) < 0.05 else '⚠️ '
    print(f"\n✓ {name:<22}  Val R²={val_r2:.4f}  Test R²={test_r2:.4f}  "
          f"Gap={gap:+.4f}  {flag}")
    
checkpoint = load_checkpoint()
completed  = checkpoint['completed_models']

# ── TEMPORARY: manually mark completed models to skip retraining ──
# Remove this block once retraining of CatBoost is confirmed complete
# already_done = ['LightGBM', 'XGBoost', 'Random Forest',
#                 'Gradient Boosting', 'MLP', 'AdaBoost', 'SVR', 'CatBoost']
# for m in already_done:
#     if m not in completed:
#         completed.append(m)
# save_checkpoint(completed)
# print(f"📌 Manually skipping: {already_done}")
# ── END TEMPORARY ──

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n" + "="*80); print("LOADING DATA"); print("="*80)

df_all = pd.read_csv(DATA_FILE, low_memory=False)
df     = df_all[df_all['System'] == SYSTEM].copy()
print(f"✓ {SYSTEM}: {len(df):,} records | {df['Manhole_ID'].nunique():,} manholes")

# ── REGRESSION FILTER ──────────────────────────────────────────────────────
# Restrict regression to records where pipe diameter was derived from
# inspection inlet/outlet data (Diameter_Source == 'Inspection_Data')
# AND water depth is valid and positive.
# This excludes the 52,269 newly added V2 records that only gained a valid
# surcharge ratio because inspection diameter filled a network data gap.
# These records are structurally different and caused R² to drop from 0.7
# to 0.5 when included. Restricting to inspection-validated records
# recovers R² while retaining the improved V2 diameter calculation.
# if 'Diameter_Source' in df.columns:
#     before = len(df)
#     df = df[
#         (df['Diameter_Source'] == 'Inspection_Data') &
#         (df['Water_Depth'].notna())
#         # Water_Depth > 0 removed — includes dry manholes
#     ].copy()
#     after = len(df)
#     print(f"✓ Regression filter applied:")
#     print(f"    Before:   {before:,} records")
#     print(f"    After:    {after:,} records")
#     print(f"    Excluded: {before-after:,} records "
#           f"(network fallback diameter or null water depth)")
# else:
#     print("⚠ Diameter_Source column not found — no regression filter applied")
#     print("  Make sure you are using the V2 file")
# ── END REGRESSION FILTER ──────────────────────────────────────────────────

avail  = [f for f in ALL_FEATURES if f in df.columns]
cats   = [f for f in CATEGORICAL_FEATURES if f in avail]
if 'Silt_Depth' in avail: avail.remove('Silt_Depth')

X = df[avail].copy(); y = df[TARGET].copy()
mask = y.notna(); X, y, df = X[mask], y[mask], df[mask]
for col in avail:
    if X[col].isna().any():
        X[col] = X[col].fillna('Unknown') if col in cats else X[col].fillna(X[col].median())

print(f"✓ Clean: {len(X):,} rows | {len(avail)} features")
# ==============================================================================
# MANHOLE-BASED SPLIT
# ==============================================================================

unique_mh = df['Manhole_ID'].unique()
tv_mh, te_mh = train_test_split(unique_mh, test_size=TEST_FRAC, random_state=RANDOM_STATE)
tr_mh, va_mh  = train_test_split(tv_mh, test_size=VAL_FRAC/(1-TEST_FRAC), random_state=RANDOM_STATE)

tr_m = df['Manhole_ID'].isin(tr_mh)
va_m = df['Manhole_ID'].isin(va_mh)
te_m = df['Manhole_ID'].isin(te_mh)

X_train, y_train = X[tr_m].reset_index(drop=True), y[tr_m].reset_index(drop=True)
X_val,   y_val   = X[va_m].reset_index(drop=True), y[va_m].reset_index(drop=True)
X_test,  y_test  = X[te_m].reset_index(drop=True), y[te_m].reset_index(drop=True)

assert not (set(tr_mh) & set(te_mh)), "Train/Test overlap!"
print(f"\n  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print("✅ No manhole overlap")

checkpoint   = load_checkpoint()
completed    = checkpoint['completed_models']
start_global = datetime.now()
if completed: print(f"\n📌 Skipping already done: {', '.join(completed)}")

# ==============================================================================
# 1. LIGHTGBM
# ==============================================================================
if not SHAP_ONLY:
    if lgb is not None and 'LightGBM' not in completed:
        print("\n" + "="*80); print(f"[1/8] LIGHTGBM  ({N_FAST} trials)"); print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte = X_train.copy(), X_val.copy(), X_test.copy()
        for col in cats:
            for d in [Xtr, Xva, Xte]: d[col] = d[col].astype('category')

        def obj_lgb(trial):
            p = dict(n_estimators=trial.suggest_int('n_estimators',100,1000),
                     max_depth=trial.suggest_int('max_depth',3,15),
                     learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
                     num_leaves=trial.suggest_int('num_leaves',20,150),
                     min_child_samples=trial.suggest_int('min_child_samples',5,100),
                     subsample=trial.suggest_float('subsample',0.5,1.0),
                     colsample_bytree=trial.suggest_float('colsample_bytree',0.5,1.0),
                     reg_alpha=trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
                     reg_lambda=trial.suggest_float('reg_lambda',1e-8,10.0,log=True),
                     random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
            r2 = r2_score(y_val, lgb.LGBMRegressor(**p).fit(Xtr,y_train).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2})
            if trial.number%10==0: print(f"  Trial {trial.number:>3}/{N_FAST}: R²={r2:.4f}")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_lgb, n_trials=N_FAST)
        bp = {**st.best_params,'n_jobs':-1,'verbose':-1,'random_state':RANDOM_STATE}
        Xtv = pd.concat([Xtr, Xva])
        for col in cats:
            Xtv[col] = Xtv[col].astype('category')
        fm = lgb.LGBMRegressor(**bp).fit(Xtv, pd.concat([y_train, y_val]))
        yp = np.clip(fm.predict(Xte),0,None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds()}
        print_result('LightGBM', st.best_value, m['R²'])
        save_results('LightGBM', fm, bp, m, hist)
        completed.append('LightGBM'); save_checkpoint(completed)

    # ==============================================================================
    # 2. XGBOOST
    # ==============================================================================

    if xgb is not None and 'XGBoost' not in completed:
        print("\n" + "="*80); print(f"[2/8] XGBOOST  ({N_FAST} trials)"); print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte, _ = encode_cats_numeric(X_train, X_val, X_test, cats)

        def obj_xgb(trial):
            p = dict(n_estimators=trial.suggest_int('n_estimators',100,1000),
                     max_depth=trial.suggest_int('max_depth',3,15),
                     learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
                     subsample=trial.suggest_float('subsample',0.5,1.0),
                     colsample_bytree=trial.suggest_float('colsample_bytree',0.5,1.0),
                     gamma=trial.suggest_float('gamma',1e-8,1.0,log=True),
                     reg_alpha=trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
                     reg_lambda=trial.suggest_float('reg_lambda',1e-8,10.0,log=True),
                     min_child_weight=trial.suggest_int('min_child_weight',1,10),
                     random_state=RANDOM_STATE, n_jobs=-1)
            r2 = r2_score(y_val, xgb.XGBRegressor(**p).fit(Xtr,y_train).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2})
            if trial.number%10==0: print(f"  Trial {trial.number:>3}/{N_FAST}: R²={r2:.4f}")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_xgb, n_trials=N_FAST)
        bp = {**st.best_params,'n_jobs':-1,'random_state':RANDOM_STATE}
        fm = xgb.XGBRegressor(**bp).fit(pd.concat([Xtr,Xva]), pd.concat([y_train,y_val]))
        yp = np.clip(fm.predict(Xte),0,None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds()}
        print_result('XGBoost', st.best_value, m['R²'])
        save_results('XGBoost', fm, bp, m, hist)
        completed.append('XGBoost'); save_checkpoint(completed)

    # ==============================================================================
    # 3. CATBOOST
    # ==============================================================================

    if cb is not None and 'CatBoost' not in completed:
        print("\n" + "="*80); print(f"[3/8] CATBOOST  ({N_MEDIUM} trials)"); print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte = X_train.copy(), X_val.copy(), X_test.copy()
        for col in cats:
            for d in [Xtr,Xva,Xte]: d[col] = d[col].astype(str)

        def obj_cat(trial):
            p = dict(iterations=trial.suggest_int('iterations',100,1000),
                     depth=trial.suggest_int('depth',3,10),
                     learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
                     l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1e-8,10.0,log=True),
                     subsample=trial.suggest_float('subsample',0.5,1.0),
                     cat_features=cats, random_state=RANDOM_STATE,
                     verbose=False, task_type='CPU', thread_count=-1)
            r2 = r2_score(y_val, cb.CatBoostRegressor(**p).fit(Xtr,y_train,verbose=False).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2})
            if trial.number%10==0: print(f"  Trial {trial.number:>3}/{N_MEDIUM}: R²={r2:.4f}")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_cat, n_trials=N_MEDIUM)
        bp = {**st.best_params,'cat_features':cats,'thread_count':-1,'random_state':RANDOM_STATE}
        fm = cb.CatBoostRegressor(**bp).fit(pd.concat([Xtr,Xva]), pd.concat([y_train,y_val]),verbose=False)
        yp = np.clip(fm.predict(Xte),0,None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds()}
        print_result('CatBoost', st.best_value, m['R²'])
        save_results('CatBoost', fm, bp, m, hist)
        completed.append('CatBoost'); save_checkpoint(completed)

    # ==============================================================================
    # 4. RANDOM FOREST
    # ==============================================================================

    if 'Random Forest' not in completed:
        print("\n" + "="*80); print(f"[4/8] RANDOM FOREST  ({N_MEDIUM} trials)"); print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte, _ = encode_cats_numeric(X_train, X_val, X_test, cats)

        def obj_rf(trial):
            p = dict(n_estimators=trial.suggest_int('n_estimators',100,500),
                     max_depth=trial.suggest_int('max_depth',10,50),
                     min_samples_split=trial.suggest_int('min_samples_split',2,20),
                     min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
                     max_features=trial.suggest_categorical('max_features',['sqrt','log2',0.5,0.7]),
                     random_state=RANDOM_STATE, n_jobs=-1)
            r2 = r2_score(y_val, RandomForestRegressor(**p).fit(Xtr,y_train).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2})
            if trial.number%5==0: print(f"  Trial {trial.number:>3}/{N_MEDIUM}: R²={r2:.4f}")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_rf, n_trials=N_MEDIUM)
        bp = {**st.best_params,'n_jobs':-1,'random_state':RANDOM_STATE}
        fm = RandomForestRegressor(**bp).fit(pd.concat([Xtr,Xva]), pd.concat([y_train,y_val]))
        yp = np.clip(fm.predict(Xte),0,None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds()}
        print_result('Random Forest', st.best_value, m['R²'])
        save_results('Random Forest', fm, bp, m, hist)
        completed.append('Random Forest'); save_checkpoint(completed)

    # ==============================================================================
    # 5. GRADIENT BOOSTING
    # ==============================================================================

    if 'Gradient Boosting' not in completed:
        print("\n" + "="*80); print(f"[5/8] GRADIENT BOOSTING  ({N_SLOW} trials)"); print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte, _ = encode_cats_numeric(X_train, X_val, X_test, cats)

        def obj_gb(trial):
            p = dict(n_estimators=trial.suggest_int('n_estimators',100,500),
                     max_depth=trial.suggest_int('max_depth',3,10),
                     learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
                     min_samples_split=trial.suggest_int('min_samples_split',2,20),
                     min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
                     subsample=trial.suggest_float('subsample',0.5,1.0),
                     max_features=trial.suggest_categorical('max_features',['sqrt','log2',None]),
                     random_state=RANDOM_STATE)
            r2 = r2_score(y_val, GradientBoostingRegressor(**p).fit(Xtr,y_train).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2})
            if trial.number%3==0:
                elapsed=(datetime.now()-t0).total_seconds()
                eta=(elapsed/(trial.number+1))*(N_SLOW-trial.number-1)
                print(f"  Trial {trial.number:>3}/{N_SLOW}: R²={r2:.4f} | ETA {eta/60:.1f}min")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_gb, n_trials=N_SLOW)
        bp = {**st.best_params,'random_state':RANDOM_STATE}
        fm = GradientBoostingRegressor(**bp).fit(pd.concat([Xtr,Xva]), pd.concat([y_train,y_val]))
        yp = np.clip(fm.predict(Xte),0,None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds()}
        print_result('Gradient Boosting', st.best_value, m['R²'])
        save_results('Gradient Boosting', fm, bp, m, hist)
        completed.append('Gradient Boosting'); save_checkpoint(completed)

    # ==============================================================================
    # 6. MLP NEURAL NETWORK
    # ==============================================================================

    if 'MLP' not in completed:
        print("\n" + "="*80); print(f"[6/8] MLP NEURAL NETWORK  ({N_SLOW} trials)"); print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte, _ = encode_cats_numeric(X_train, X_val, X_test, cats)
        Xtr, Xva, Xte, sc = scale_numeric(Xtr, Xva, Xte, cats)

        def obj_mlp(trial):
            n_layers = trial.suggest_int('n_layers', 1, 4)
            layer_sz = trial.suggest_categorical('layer_size', [64, 128, 256, 512])
            layers   = tuple([layer_sz] * n_layers)
            p = dict(hidden_layer_sizes=layers,
                     activation=trial.suggest_categorical('activation', ['relu', 'tanh']),
                     alpha=trial.suggest_float('alpha', 1e-5, 1.0, log=True),
                     learning_rate_init=trial.suggest_float('learning_rate_init', 1e-4, 0.05, log=True),
                     batch_size=trial.suggest_categorical('batch_size', [128, 256, 512]),
                     max_iter=200, early_stopping=True, validation_fraction=0.1,
                     n_iter_no_change=15, random_state=RANDOM_STATE)
            r2 = r2_score(y_val, MLPRegressor(**p).fit(Xtr,y_train).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2,'layers':str(layers)})
            if trial.number%10==0: print(f"  Trial {trial.number:>3}/{N_SLOW}: R²={r2:.4f}  layers={layers}")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_mlp, n_trials=N_SLOW)
        best = st.best_params
        n_layers = best.pop('n_layers'); layer_sz = best.pop('layer_size')
        bp = {**best, 'hidden_layer_sizes': tuple([layer_sz]*n_layers),
              'max_iter': 500, 'early_stopping': True, 'validation_fraction': 0.1,
              'n_iter_no_change': 20, 'random_state': RANDOM_STATE}
        fm = MLPRegressor(**bp).fit(pd.concat([Xtr,Xva]), pd.concat([y_train,y_val]))
        yp = np.clip(fm.predict(Xte), 0, None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds()}
        print_result('MLP', st.best_value, m['R²'])
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(OUTPUT_DIR, f'MLP_{ts_str}.pkl'), 'wb') as f:
            pickle.dump({'model': fm, 'scaler': sc, 'model_name': 'MLP',
                         'best_params': bp, 'metrics': m}, f)
        save_results('MLP', fm, bp, m, hist)
        completed.append('MLP'); save_checkpoint(completed)

    # ==============================================================================
    # 7. ADABOOST
    # ==============================================================================

    if 'AdaBoost' not in completed:
        print("\n" + "="*80); print(f"[7/8] ADABOOST  ({N_MEDIUM} trials)"); print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte, _ = encode_cats_numeric(X_train, X_val, X_test, cats)

        def obj_ada(trial):
            base_depth = trial.suggest_int('base_max_depth', 2, 4)
            p = dict(n_estimators=trial.suggest_int('n_estimators', 50, 500),
                     learning_rate=trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
                     loss=trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
                     estimator=DecisionTreeRegressor(max_depth=base_depth, random_state=RANDOM_STATE),
                     random_state=RANDOM_STATE)
            r2 = r2_score(y_val, AdaBoostRegressor(**p).fit(Xtr,y_train).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2,'base_depth':base_depth})
            if trial.number%5==0: print(f"  Trial {trial.number:>3}/{N_MEDIUM}: R²={r2:.4f}  base_depth={base_depth}")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_ada, n_trials=N_MEDIUM)
        best = st.best_params
        base_depth = best.pop('base_max_depth')
        bp = {**best,
              'estimator': DecisionTreeRegressor(max_depth=base_depth, random_state=RANDOM_STATE),
              'random_state': RANDOM_STATE}
        fm = AdaBoostRegressor(**bp).fit(pd.concat([Xtr,Xva]), pd.concat([y_train,y_val]))
        yp = np.clip(fm.predict(Xte), 0, None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds()}
        print_result('AdaBoost', st.best_value, m['R²'])
        save_results('AdaBoost', fm, bp, m, hist)
        completed.append('AdaBoost'); save_checkpoint(completed)

    # ==============================================================================
    # 8. SVR
    # ==============================================================================

    if 'SVR' not in completed:
        print("\n" + "="*80)
        print(f"[8/8] SVR  ({N_SVR} trials)")
        print(f"  ⚠ SVR is O(n²) — trials use {SVR_TRIAL_N:,} rows, "
              f"final model uses {SVR_FINAL_N:,} rows")
        print("="*80)
        t0 = datetime.now(); hist = []

        Xtr, Xva, Xte, _ = encode_cats_numeric(X_train, X_val, X_test, cats)
        Xtr, Xva, Xte, sc_svr = scale_numeric(Xtr, Xva, Xte, cats)
        Xtr_s, ytr_s = subsample_rows(Xtr, y_train, SVR_TRIAL_N)

        def obj_svr(trial):
            p = dict(kernel=trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                     C=trial.suggest_float('C', 0.1, 100.0, log=True),
                     epsilon=trial.suggest_float('epsilon', 1e-4, 1.0, log=True),
                     gamma=trial.suggest_categorical('gamma', ['scale', 'auto']))
            if p['kernel'] == 'poly':
                p['degree'] = trial.suggest_int('degree', 2, 4)
            r2 = r2_score(y_val, SVR(**p).fit(Xtr_s, ytr_s).predict(Xva))
            hist.append({'trial':trial.number,'r2':r2})
            if trial.number%5==0: print(f"  Trial {trial.number:>3}/{N_SVR}: R²={r2:.4f}  kernel={p['kernel']}")
            return r2

        st = optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
        st.optimize(obj_svr, n_trials=N_SVR)
        bp = st.best_params
        Xtv = pd.concat([Xtr, Xva]); ytv = pd.concat([y_train, y_val])
        Xtv_s, ytv_s = subsample_rows(Xtv, ytv, SVR_FINAL_N)
        print(f"  Training final SVR on {len(Xtv_s):,} rows …")
        fm = SVR(**bp).fit(Xtv_s, ytv_s)
        yp = np.clip(fm.predict(Xte), 0, None)
        m  = {'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
              'RMSE':float(np.sqrt(mean_squared_error(y_test,yp))),
              'training_time':(datetime.now()-t0).total_seconds(),
              'note': f'Trained on {SVR_FINAL_N} row subsample'}
        val_r2 = r2_score(y_val, fm.predict(Xva))
        print_result('SVR', val_r2, m['R²'])
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(OUTPUT_DIR, f'SVR_{ts_str}.pkl'), 'wb') as f:
            pickle.dump({'model': fm, 'scaler': sc_svr, 'model_name': 'SVR',
                         'best_params': bp, 'metrics': m}, f)
        save_results('SVR', fm, bp, m, hist)
        completed.append('SVR'); save_checkpoint(completed)

    # ==============================================================================
    # FINAL SUMMARY
    # ==============================================================================

    print("\n" + "="*80); print("TRAINING COMPLETE!"); print("="*80)
    print(f"\nTotal time: {(datetime.now()-start_global).total_seconds()/3600:.2f} hours")
    print(f"Completed : {len(completed)}/8 models")

    sf = os.path.join(OUTPUT_DIR, 'optimization_summary.csv')
    if os.path.exists(sf):
        sdf = (pd.read_csv(sf)
               .sort_values('R²', ascending=False)
               .drop_duplicates(subset='Model', keep='first')
               .reset_index(drop=True))
        print(f"\n{'='*80}")
        print("FINAL RESULTS — MANHOLE-BASED SPLIT (No Interactions, No Silt_Depth)")
        print(f"{'='*80}")
        print(f"\n{'Rank':<5} {'Model':<25} {'R²':>8} {'MAE':>8} {'RMSE':>8} {'Time':>10}")
        print("-"*65)
        medals = {0:'🏆', 1:'🥈', 2:'🥉'}
        for i, (_, row) in enumerate(sdf.iterrows()):
            sym = medals.get(i, '   ')
            print(f"{sym} {i+1:<3} {row['Model']:<25} {row['R²']:>8.4f} "
                  f"{row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
                  f"{row['Training_Time_s']/60:>8.1f}min")
        best = sdf.iloc[0]
        print(f"\n🏆 BEST: {best['Model']}  R²={best['R²']:.4f}")

        colors = ['gold' if i==0 else '#3498db' for i in range(len(sdf))]
        fig, ax = plt.subplots(figsize=(13, 7))
        ax.barh(range(len(sdf)), sdf['R²'], color=colors, alpha=0.85)
        ax.set_yticks(range(len(sdf))); ax.set_yticklabels(sdf['Model'], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Test R²', fontsize=12, fontweight='bold')
        ax.set_title('8-Model Comparison — Manhole-Based Split\nNo Interactions | No Silt_Depth',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (_, row) in enumerate(sdf.iterrows()):
            ax.text(row['R²']+0.003, i, f"{row['R²']:.4f}", va='center', fontweight='bold', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison_8models.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ model_comparison_8models.png")

    print(f"\n📁 {OUTPUT_DIR}")
    print(f"✅ TRAINING DONE\n{'='*80}")

    if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)

# ==============================================================================
# SHAP — TOP-2 MODELS
# ==============================================================================

try:
    import shap
    print(f"\n✓ SHAP {shap.__version__}  — starting top-2 analysis")
except ImportError:
    raise SystemExit("❌ pip install shap  then re-run")

import glob, matplotlib.patches as mpatches

SHAP_MAX_ROWS       = 5000
KERNEL_BG_ROWS      = 500
KERNEL_EXPLAIN_ROWS = 1000

CATEGORY_COLORS = {
    'Maintenance/Defects': '#e74c3c', 'Climate/Weather':  '#3498db',
    'Pipe Properties':     '#2ecc71', 'Topographic':       '#9b59b6',
    'Land Use / Urban':    '#f39c12', 'Manhole Config':    '#1abc9c',
    'Silt/Capacity':       '#e67e22', 'Other':             '#95a5a6',
}

def classify_feature(f):
    fl = f.lower()
    if any(k in fl for k in ['silt','capacity']):                       return 'Silt/Capacity'
    if any(k in fl for k in ['rain','temp','humid','wind','dewpoint']):  return 'Climate/Weather'
    if any(k in fl for k in ['pipe','upstream','downstream','slope',
                             'diameter','length','age_avg']):           return 'Pipe Properties'
    if any(k in fl for k in ['elev','curvat']):                         return 'Topographic'
    if any(k in fl for k in ['bldg','pop','catch','imperv','aadt',
                             'landuse']):                               return 'Land Use / Urban'
    if any(k in fl for k in ['num_inlet','num_upstream','num_downstream',
                             'reservoir','prop_manhole','inspection_year',
                             'inspection_month','age_at','size_of_cover',
                             'type','geol']):                           return 'Manhole Config'
    if any(k in fl for k in ['clean','maintenance','crack','defect','deform',
                             'infiltr','corros','environ','inspect','broken',
                             'connect','flow','block','general']):      return 'Maintenance/Defects'
    return 'Other'

def prepare_data_for_model(model_name, X_tr, X_va, X_te, cat_cols):
    name = model_name.lower()
    if 'lightgbm' in name:
        Xtr, Xva, Xte = X_tr.copy(), X_va.copy(), X_te.copy()
        for col in cat_cols:
            for d in [Xtr, Xva, Xte]: d[col] = d[col].astype('category')
        return Xtr, Xva, Xte, None
    if name == 'catboost':
        Xtr, Xva, Xte = X_tr.copy(), X_va.copy(), X_te.copy()
        for col in cat_cols:
            for d in [Xtr, Xva, Xte]: d[col] = d[col].astype(str)
        return Xtr, Xva, Xte, None
    if name in ('mlp', 'svr'):
        Xtr, Xva, Xte, _ = encode_cats_numeric(X_tr, X_va, X_te, cat_cols)
        Xtr, Xva, Xte, sc = scale_numeric(Xtr, Xva, Xte, cat_cols)
        return Xtr, Xva, Xte, sc
    Xtr, Xva, Xte, _ = encode_cats_numeric(X_tr, X_va, X_te, cat_cols)
    return Xtr, Xva, Xte, None

def find_pkl(model_name):
    pattern = os.path.join(OUTPUT_DIR, f"{model_name.replace(' ','_')}_*.pkl")
    files   = [f for f in glob.glob(pattern)
               if not any(x in f for x in ['params','trials'])]
    if not files:
        raise FileNotFoundError(f"No pkl found for '{model_name}': {pattern}")
    return sorted(files, key=os.path.getmtime)[-1]

def run_shap_suite(model_name, model, X_tr, X_te, y_te, is_tree=True):
    out = os.path.join(OUTPUT_DIR, f"SHAP_{model_name.replace(' ','_')}")
    os.makedirs(out, exist_ok=True)
    feat_names = list(X_te.columns)

    print(f"\n{'─'*70}")
    print(f"  SHAP: {model_name}  ({'TreeExplainer' if is_tree else 'KernelExplainer'})")
    print(f"{'─'*70}")

    y_check  = np.clip(model.predict(X_te), 0, None)
    r2_check = r2_score(y_te, y_check)
    print(f"  Sanity R² (full test set): {r2_check:.4f}")
    if r2_check < 0:
        raise ValueError(f"❌ Negative R² for {model_name} — data prep mismatch")

    t0 = datetime.now()
    if is_tree:
        n   = min(SHAP_MAX_ROWS, len(X_te))
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_te), n, replace=False)
        X_s = X_te.iloc[idx].reset_index(drop=True)
        y_s = y_te.iloc[idx].reset_index(drop=True)
        explainer = shap.TreeExplainer(model)
        sv_arr    = np.array(explainer.shap_values(X_s))
        base_val  = (float(explainer.expected_value)
                     if np.isscalar(explainer.expected_value)
                     else float(explainer.expected_value[0]))
    else:
        X_bg   = shap.sample(X_tr, KERNEL_BG_ROWS, random_state=RANDOM_STATE)
        ex_idx = np.random.default_rng(RANDOM_STATE+1).choice(len(X_te),
                                                              min(KERNEL_EXPLAIN_ROWS, len(X_te)), replace=False)
        X_s    = X_te.iloc[ex_idx].reset_index(drop=True)
        y_s    = y_te.iloc[ex_idx].reset_index(drop=True)
        print(f"  KernelExplainer: bg={KERNEL_BG_ROWS} rows, explain={len(X_s)} rows …")
        explainer = shap.KernelExplainer(model.predict, X_bg)
        sv_arr    = np.array(explainer.shap_values(X_s))
        base_val  = float(explainer.expected_value)

    print(f"  SHAP values computed in {(datetime.now()-t0).total_seconds():.1f}s")

    mean_abs = np.abs(sv_arr).mean(axis=0)
    imp_df   = pd.DataFrame({
        'Feature':       feat_names,
        'Mean_Abs_SHAP': mean_abs,
        'Category':      [classify_feature(f) for f in feat_names]
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    imp_df['Pct'] = imp_df['Mean_Abs_SHAP'] / imp_df['Mean_Abs_SHAP'].sum() * 100
    imp_df.to_csv(os.path.join(out, 'feature_importance.csv'), index=False)

    # 1. Summary Bar
    top40  = imp_df.head(40)
    colors = [CATEGORY_COLORS.get(c, '#95a5a6') for c in top40['Category']]
    fig, ax = plt.subplots(figsize=(12, max(8, len(top40)*0.32)))
    ax.barh(range(len(top40)), top40['Mean_Abs_SHAP'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top40))); ax.set_yticklabels(top40['Feature'], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel('Mean |SHAP value|', fontweight='bold')
    ax.set_title(f'{model_name} — SHAP Feature Importance (Top 40)\nTest R²={r2_check:.4f}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    mx = top40['Mean_Abs_SHAP'].max()
    for i, (v, p) in enumerate(zip(top40['Mean_Abs_SHAP'], top40['Pct'])):
        ax.text(v+mx*0.005, i, f'{p:.1f}%', va='center', fontsize=7)
    patches = [mpatches.Patch(color=v, label=k) for k,v in CATEGORY_COLORS.items()
               if k in top40['Category'].values]
    ax.legend(handles=patches, loc='lower right', fontsize=7, title='Category')
    plt.tight_layout()
    plt.savefig(os.path.join(out,'summary_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ summary_bar.png")

    # 2. Beeswarm
    shap_expl = shap.Explanation(values=sv_arr, base_values=np.full(len(X_s), base_val),
                                 data=X_s.values, feature_names=feat_names)
    fig, _ = plt.subplots(figsize=(12, 10))
    shap.plots.beeswarm(shap_expl, max_display=25, show=False)
    plt.title(f'{model_name} — SHAP Beeswarm (Top 25)\nTest R²={r2_check:.4f}',
              fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out,'beeswarm.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ beeswarm.png")

    # 3. Waterfall — highest predicted surcharge
    y_pred_s = np.clip(model.predict(X_s), 0, None)
    idx_high = int(np.argmax(y_pred_s))
    expl_one = shap.Explanation(values=sv_arr[idx_high], base_values=base_val,
                                data=X_s.iloc[idx_high].values, feature_names=feat_names)
    fig, _ = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(expl_one, max_display=20, show=False)
    plt.title(f'{model_name} — Waterfall: Highest Predicted Surcharge\n'
              f'True={float(y_s.iloc[idx_high]):.4f}  Pred={y_pred_s[idx_high]:.4f}',
              fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out,'waterfall_sample.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ waterfall_sample.png")

    # 4. Dependence plots (top 6)
    top6 = imp_df['Feature'].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, feat in enumerate(top6):
        ax  = axes.flatten()[i]; fi = feat_names.index(feat)
        
        # Handle categorical features — encode to numeric for plotting
        raw = X_s[feat]
        if raw.dtype == object or str(raw.dtype) == 'category':
            fv = pd.Categorical(raw).codes.astype(float)
            fv[fv < 0] = np.nan  # -1 codes for NaN categories
        else:
            try:
                fv = raw.values.astype(float)
            except (ValueError, TypeError):
                fv = pd.Categorical(raw).codes.astype(float)
        sc_ = axes.flatten()[i].scatter(fv, sv_arr[:,fi], c=sv_arr[:,fi],
                                        cmap='RdBu_r', alpha=0.4, s=8, rasterized=True)
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel(feat, fontsize=9); ax.set_ylabel('SHAP value', fontsize=9)
        rank = imp_df.index[imp_df['Feature']==feat][0]+1
        ax.set_title(f'#{rank} {feat}\n[{classify_feature(feat)}]', fontsize=9, fontweight='bold')
        plt.colorbar(sc_, ax=ax, label='SHAP', pad=0.02); ax.grid(alpha=0.2)
    plt.suptitle(f'{model_name} — SHAP Dependence (Top 6)', fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out,'dependence_plots.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ dependence_plots.png")

    # 5. Interaction heatmap (tree models only)
    if is_tree:
        top10 = imp_df['Feature'].head(10).tolist()
        try:
            Xtr_t10 = X_tr[top10].copy()
            for col in cats:
                if col in top10: Xtr_t10[col] = Xtr_t10[col].astype('category')
            X_s_t10 = X_s[top10].copy()
            for col in cats:
                if col in top10: X_s_t10[col] = X_s_t10[col].astype('category')
            mini = lgb.LGBMRegressor(n_estimators=200, max_depth=6, num_leaves=31,
                                     random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
            mini.fit(Xtr_t10, y_train)
            me    = shap.TreeExplainer(mini)
            sv_ix = me.shap_interaction_values(X_s_t10)
            mat   = np.abs(sv_ix).mean(axis=0).copy(); np.fill_diagonal(mat, 0)
            pd.DataFrame(mat, index=top10, columns=top10).to_csv(os.path.join(out,'interactions.csv'))
            fig, ax = plt.subplots(figsize=(11, 9))
            im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(10)); ax.set_yticks(range(10))
            ax.set_xticklabels(top10, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(top10, fontsize=8)
            plt.colorbar(im, ax=ax, label='Mean |SHAP interaction|')
            ax.set_title(f'{model_name} — SHAP Interaction Heatmap (Top 10)', fontweight='bold', fontsize=12)
            for i2 in range(10):
                for j2 in range(10):
                    if i2!=j2 and mat[i2,j2]>0:
                        ax.text(j2, i2, f'{mat[i2,j2]:.4f}', ha='center', va='center', fontsize=5.5,
                                color='white' if mat[i2,j2]>mat.max()*0.65 else 'black')
            plt.tight_layout()
            plt.savefig(os.path.join(out,'interactions.png'), dpi=300, bbox_inches='tight')
            plt.close(); print("  ✓ interactions.png  |  interactions.csv")
        except Exception as e:
            print(f"  ⚠ Interactions skipped: {e}")
    else:
        print("  ℹ Interaction heatmap skipped for non-tree model")

    # 6. Category summary
    cat_imp = imp_df.groupby('Category')['Mean_Abs_SHAP'].sum().sort_values(ascending=False)
    total   = cat_imp.sum()
    ccols   = [CATEGORY_COLORS.get(c,'#95a5a6') for c in cat_imp.index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.barh(range(len(cat_imp)), cat_imp.values, color=ccols, alpha=0.85)
    ax1.set_yticks(range(len(cat_imp))); ax1.set_yticklabels(cat_imp.index, fontsize=10)
    ax1.invert_yaxis(); ax1.set_xlabel('Sum of Mean |SHAP|', fontweight='bold')
    ax1.set_title('Category Importance', fontweight='bold'); ax1.grid(axis='x', alpha=0.3)
    for i,(v,p) in enumerate(zip(cat_imp.values, cat_imp.values/total*100)):
        ax1.text(v+total*0.005, i, f'{p:.1f}%', va='center', fontsize=9)
    wedges, texts, autos = ax2.pie(cat_imp.values, labels=cat_imp.index, colors=ccols,
                                   autopct='%1.1f%%', startangle=140, pctdistance=0.75)
    for t in texts:  t.set_fontsize(9)
    for t in autos:  t.set_fontsize(8)
    ax2.set_title('Proportion of Total SHAP', fontweight='bold')
    plt.suptitle(f'{model_name} — SHAP by Feature Category | Test R²={r2_check:.4f}',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out,'category_summary.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  ✓ category_summary.png")

    print(f"\n  📊 Top-10 SHAP features for {model_name}:")
    print(f"  {'Rank':<5} {'Feature':<45} {'Mean|SHAP|':>12}  {'%':>6}  Category")
    print("  " + "─"*80)
    for _, row in imp_df.head(10).iterrows():
        print(f"  {_+1:<5} {row['Feature']:<45} {row['Mean_Abs_SHAP']:>12.6f}"
              f"  {row['Pct']:>5.1f}%  {row['Category']}")
    n80 = int((imp_df['Pct'].cumsum()<=80).sum())+1
    print(f"\n  → {n80} features explain 80% of SHAP impact")
    return imp_df

TREE_MODELS = {'lightgbm', 'xgboost', 'catboost', 'random forest',
               'gradient boosting', 'adaboost'}
def is_tree_model(name): return name.lower() in TREE_MODELS

sf = os.path.join(OUTPUT_DIR, 'optimization_summary.csv')
if not os.path.exists(sf):
    raise SystemExit("❌ optimization_summary.csv not found — run training first")

sdf  = (pd.read_csv(sf)
        .sort_values('R²', ascending=False)
        .drop_duplicates(subset='Model', keep='first')  # ← keep best run per model
        .reset_index(drop=True))
top2 = sdf['Model'].head(2).tolist()

print(f"\n{'='*80}"); print("TOP-2 MODELS SELECTED FOR SHAP"); print(f"{'='*80}")
for i, name in enumerate(top2):
    r2 = sdf.loc[sdf['Model']==name,'R²'].values[0]
    print(f"  #{i+1}  {name:<25}  R²={r2:.4f}")

imp_dfs = {}
for model_name in top2:
    print(f"\n{'='*80}"); print(f"SHAP — {model_name}"); print(f"{'='*80}")
    pkl_path = find_pkl(model_name)
    print(f"  Loading: {pkl_path}")
    with open(pkl_path,'rb') as f: pkg = pickle.load(f)
    model = pkg['model']
    Xtr_m, Xva_m, Xte_m, _ = prepare_data_for_model(model_name, X_train, X_val, X_test, cats)
    imp_dfs[model_name] = run_shap_suite(
        model_name=model_name, model=model,
        X_tr=Xtr_m, X_te=Xte_m, y_te=y_test,
        is_tree=is_tree_model(model_name))

# Side-by-side comparison
if len(imp_dfs) == 2:
    names = list(imp_dfs.keys())
    df1   = imp_dfs[names[0]][['Feature','Mean_Abs_SHAP','Category']]
    df2   = imp_dfs[names[1]][['Feature','Mean_Abs_SHAP']]
    mg    = df1.merge(df2, on='Feature', suffixes=('_1','_2'))
    mg['n1'] = mg['Mean_Abs_SHAP_1'] / mg['Mean_Abs_SHAP_1'].sum()
    mg['n2'] = mg['Mean_Abs_SHAP_2'] / mg['Mean_Abs_SHAP_2'].sum()
    mg = mg.assign(avg=(mg['n1']+mg['n2'])/2).sort_values('avg',ascending=False).head(20).reset_index(drop=True)
    r2_1 = sdf.loc[sdf['Model']==names[0],'R²'].values[0]
    r2_2 = sdf.loc[sdf['Model']==names[1],'R²'].values[0]
    x = np.arange(len(mg)); w = 0.35
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(x-w/2, mg['n1'], w, label=f'{names[0]} (R²={r2_1:.4f})', color='#2ecc71', alpha=0.85)
    ax.bar(x+w/2, mg['n2'], w, label=f'{names[1]} (R²={r2_2:.4f})', color='#3498db', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(mg['Feature'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Normalised Mean |SHAP|', fontweight='bold')
    ax.set_title(f'Top-2 Models — SHAP Comparison (Top 20 Features)\n'
                 f'{names[0]} vs {names[1]}  |  Manhole-Based Split',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_top2_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(); print(f"\n✓ shap_top2_comparison.png")

print(f"\n{'='*80}"); print("SHAP COMPLETE"); print(f"{'='*80}")
for name in top2:
    d = os.path.join(OUTPUT_DIR, f"SHAP_{name.replace(' ','_')}")
    print(f"\n  📁 {d}")
    print(f"       summary_bar.png | beeswarm.png | waterfall_sample.png")
    print(f"       dependence_plots.png | interactions.png | category_summary.png")
    print(f"       feature_importance.csv | interactions.csv")
print(f"\n  📊 shap_top2_comparison.png")
print(f"\n✅ ALL DONE\n{'='*80}")