#!/usr/bin/env python3
"""
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
from config import Paths, Params

holdout_spatialB_regression.py
==============================
DESIGN B (deployability variant) for the SEVERITY/REGRESSION tier.

Companion to holdout_spatial_regression.py (Design A). See the classification
B script header for the full A-vs-B rationale. In short:

  A (param-reuse)  : isolates spatial distribution shift.
  B (this script)  : re-tunes per source region on a 3-way DISTRICT split,
                     answering deployability to unseen districts.

3-way DISTRICT split (whole districts only):
    TRAIN 60%  -> fit each trial
    VAL   20%  -> Optuna optimises against this (cross-district signal)
    TEST  20%  -> evaluated ONCE at the end (never seen in fit or tuning)

Top-2 severity models only: LightGBM, Gradient Boosting.
Objective = R^2 on the validation districts (matches the regression baseline,
which optimised R^2). Search spaces mirror the standard baseline ranges.

MULTI_SEED True by default — a single 60/20/20 district draw with re-tuning is
the most fragile number in the whole study.
"""

import pandas as pd, numpy as np, os, json
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================ CONFIG ============================
# All paths resolved from config.py at the repository root.
# Edit INPUT_ROOT and OUTPUT_ROOT there — do not edit paths here.
SYSTEM      = Params.SYSTEM   # 'Stormwater'
DATA_FILE    = str(Paths.FINAL_REGRESSION)
OUTPUT_DIR   = str(Paths.ML_SEVERITY_SPATIAL_B)

RANDOM_STATE = 42
SYSTEM      = Params.SYSTEM
TARGET   = 'Surcharge_Ratio'
DISTRICT_COL = 'District_Insp'
VAL_DISTRICT_FRAC  = 0.20
TEST_DISTRICT_FRAC = 0.20
MODELS = ['LightGBM', 'Gradient Boosting']     # top-2 severity models
N_TRIALS_LGB = 100         # matches baseline LGB trial budget
N_TRIALS_GB  = 15          # GB is slow; matches baseline N_SLOW
MULTI_SEED = True
SEEDS = [42, 1, 7, 123, 2024]
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAINTENANCE_FEATURES = sorted(['Blockages_SinceLastInsp','General_Cleaning_SinceLastInsp',
    'Total_Maintenance_Events_SinceLastInsp','Inspection_SinceLastInsp'])
PIPE_FEATURES = sorted(['Downstream_Pipe_Age_Avg','Downstream_Pipe_Diameter_Avg','Upstream_Pipe_Diameter_Avg',
    'Downstream_Pipe_Length_Avg','Upstream_Pipe_Length_Avg','Downstream_Pipe_Slope_Avg','Upstream_Pipe_Slope_Avg'])
TOPOGRAPHIC_FEATURES=sorted(['Elev']); LANDUSE_FEATURES=sorted(['PopDensity_m2','AADT'])
MANHOLE_CONFIG_FEATURES=sorted(['Num_Inlet_Pipes','Reservoir_Distance'])
WATER_SILT_CLIMATE_FEATURES=sorted(['Silt_Coverage_Ratio','Temp_Mean_7d_Avg','Rain_7d_Sum','Windspeed_Day','Dewpoint_Day'])
CATEGORICAL_FEATURES=sorted(['Geol_Type'])
_seen=set(); ALL_FEATURES=[]
for f in (MAINTENANCE_FEATURES+PIPE_FEATURES+TOPOGRAPHIC_FEATURES+LANDUSE_FEATURES+
          MANHOLE_CONFIG_FEATURES+WATER_SILT_CLIMATE_FEATURES+CATEGORICAL_FEATURES):
    if f not in _seen: ALL_FEATURES.append(f); _seen.add(f)

def encode_cats(Xtr, Xte, cats):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for col in cats:
        ts = Xtr[col].astype(str).str.strip()
        le = LabelEncoder().fit(ts)
        Xtr[col] = le.transform(ts)
        vs = Xte[col].astype(str).str.strip()
        vs[~vs.isin(le.classes_)] = ts.value_counts().index[0]
        Xte[col] = le.transform(vs)
    return Xtr, Xte

def tune_lightgbm(Xtr, ytr, Xva, yva, cats, seed):
    Xtr2, Xva2 = Xtr.copy(), Xva.copy()
    for c in cats: Xtr2[c]=Xtr2[c].astype('category'); Xva2[c]=Xva2[c].astype('category')
    def obj(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,1000),
               max_depth=trial.suggest_int('max_depth',3,15),
               learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
               num_leaves=trial.suggest_int('num_leaves',20,150),
               min_child_samples=trial.suggest_int('min_child_samples',5,100),
               subsample=trial.suggest_float('subsample',0.5,1.0),
               colsample_bytree=trial.suggest_float('colsample_bytree',0.5,1.0),
               reg_alpha=trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
               reg_lambda=trial.suggest_float('reg_lambda',1e-8,10.0,log=True))
        m=lgb.LGBMRegressor(**p,random_state=RANDOM_STATE,n_jobs=-1,verbose=-1).fit(Xtr2,ytr)
        return r2_score(yva, m.predict(Xva2))
    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=seed),pruner=MedianPruner())
    st.optimize(obj,n_trials=N_TRIALS_LGB); return st.best_params

def tune_gb(Xtr, ytr, Xva, yva, cats, seed):
    Xtr2, Xva2 = encode_cats(Xtr, Xva, cats)
    def obj(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,500),
               max_depth=trial.suggest_int('max_depth',3,10),
               learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
               min_samples_split=trial.suggest_int('min_samples_split',2,20),
               min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
               subsample=trial.suggest_float('subsample',0.5,1.0),
               max_features=trial.suggest_categorical('max_features',['sqrt','log2',None]))
        m=GradientBoostingRegressor(**p,random_state=RANDOM_STATE).fit(Xtr2,ytr)
        return r2_score(yva, m.predict(Xva2))
    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=seed),pruner=MedianPruner())
    st.optimize(obj,n_trials=N_TRIALS_GB); return st.best_params

def final_fit_predict(name, bp, Xtr, ytr, Xte, cats):
    if name=='LightGBM':
        Xtr2,Xte2=Xtr.copy(),Xte.copy()
        for c in cats: Xtr2[c]=Xtr2[c].astype('category'); Xte2[c]=Xte2[c].astype('category')
        m=lgb.LGBMRegressor(**bp,random_state=RANDOM_STATE,n_jobs=-1,verbose=-1).fit(Xtr2,ytr)
        return m.predict(Xte2)
    if name=='Gradient Boosting':
        Xtr2,Xte2=encode_cats(Xtr,Xte,cats)
        m=GradientBoostingRegressor(**bp,random_state=RANDOM_STATE).fit(Xtr2,ytr)
        return m.predict(Xte2)

print("="*70,"\nSPATIAL HOLD-OUT — DESIGN B (re-tuned, 3-way district split)\nregression | top-2 models\n","="*70)
df=pd.read_csv(DATA_FILE,low_memory=False); df=df[df['System']==SYSTEM].copy()
avail=[f for f in ALL_FEATURES if f in df.columns]; cats=[f for f in CATEGORICAL_FEATURES if f in avail]
X=df[avail].copy(); y=df[TARGET].copy(); dist=df[DISTRICT_COL].astype(str)
mask=y.notna(); X,y,dist=X[mask].reset_index(drop=True),y[mask].reset_index(drop=True),dist[mask].reset_index(drop=True)
for col in avail:
    if X[col].isna().any(): X[col]=X[col].fillna('Unknown') if col in cats else X[col].fillna(X[col].median())

all_districts=np.array(sorted(dist.unique()))
print(f"Total districts: {len(all_districts)}  | rows: {len(X):,}  | target mean: {y.mean():.4f}")

seeds = SEEDS if MULTI_SEED else [RANDOM_STATE]
results={m:[] for m in MODELS}

for sd in seeds:
    rng=np.random.default_rng(sd)
    perm=rng.permutation(all_districts)
    n=len(perm); n_test=int(round(n*TEST_DISTRICT_FRAC)); n_val=int(round(n*VAL_DISTRICT_FRAC))
    test_d=perm[:n_test]; val_d=perm[n_test:n_test+n_val]; train_d=perm[n_test+n_val:]
    tr=dist.isin(train_d); va=dist.isin(val_d); te=dist.isin(test_d)
    X_tr,y_tr=X[tr].reset_index(drop=True),y[tr].reset_index(drop=True)
    X_va,y_va=X[va].reset_index(drop=True),y[va].reset_index(drop=True)
    X_te,y_te=X[te].reset_index(drop=True),y[te].reset_index(drop=True)
    print(f"\n[seed {sd}] districts train/val/test = {len(train_d)}/{len(val_d)}/{len(test_d)} | "
          f"rows {len(X_tr):,}/{len(X_va):,}/{len(X_te):,}")
    for name in MODELS:
        try:
            t0=datetime.now()
            bp = tune_lightgbm(X_tr,y_tr,X_va,y_va,cats,sd) if name=='LightGBM' else tune_gb(X_tr,y_tr,X_va,y_va,cats,sd)
            pred=final_fit_predict(name,bp,X_tr,y_tr,X_te,cats)
            m={'R2':r2_score(y_te,pred),'MAE':mean_absolute_error(y_te,pred),
               'RMSE':float(np.sqrt(mean_squared_error(y_te,pred))),
               'tune_seconds':(datetime.now()-t0).total_seconds()}
            results[name].append(m['R2'])
            print(f"  {name:<18} R2={m['R2']:.4f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}  ({m['tune_seconds']:.0f}s)")
            sf=os.path.join(OUTPUT_DIR,'holdout_spatialB_summary.csv')
            row=pd.DataFrame([{'Model':name,**m,'Split':'spatialB_district_60_20_20_retuned','Seed':sd,
                               'best_params':json.dumps(bp)}])
            (pd.concat([pd.read_csv(sf),row],ignore_index=True) if os.path.exists(sf) else row).to_csv(sf,index=False)
        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")

if MULTI_SEED:
    print("\n"+"="*50,"\nDESIGN-B MULTI-SEED SUMMARY (R2 mean ± std)\n","="*50)
    rows=[]
    for name in MODELS:
        if results[name]:
            arr=np.array(results[name]); rows.append({'Model':name,'R2_mean':arr.mean(),'R2_std':arr.std(),'n_seeds':len(arr)})
            print(f"  {name:<18} {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR,'holdout_spatialB_multiseed.csv'),index=False)

print(f"\nSaved -> {OUTPUT_DIR}")
print("NOTE: Design B answers DEPLOYABILITY (re-tuned per source region).")
print("      Compare against Design A (param-reuse) which isolates distribution shift.")
