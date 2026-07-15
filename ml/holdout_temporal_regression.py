#!/usr/bin/env python3
"""
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
from config import Paths, Params

holdout_temporal_regression.py
==============================
R2-04 (temporal hold-out) for the SEVERITY/REGRESSION tier.

Robustness check — NOT a replacement for the manhole-based split. Trains on
inspections from <=2018 and tests on 2019-2021, to see whether the model
generalises to FUTURE years it never saw.

Param-reuse mode: loads each model's best_params from the baseline run's
{Model}_params.json (no Optuna search). This makes the run ~1-2 hrs instead of
16+. The searched hyperparameters transfer; regression has no class-balance
terms to recompute, so reuse is clean.

The temporal split is leakage-safe by construction: maintenance features are
"events since last inspection" (already assigned only to inspections that occur
AFTER each event), so filtering training rows to <=2018 cannot pull in any
post-2018 maintenance information.
"""

import pandas as pd, numpy as np, os, json, glob, pickle
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb, xgboost as xgb, catboost as cb

# ============================ CONFIG ============================
# All paths resolved from config.py at the repository root.
# Edit INPUT_ROOT and OUTPUT_ROOT there — do not edit paths here.
SYSTEM      = Params.SYSTEM   # 'Stormwater'
DATA_FILE    = str(Paths.FINAL_REGRESSION)
OUTPUT_DIR   = str(Paths.ML_SEVERITY_TEMPORAL)
BASE_DIR     = str(Paths.ML_SEVERITY_DIR)

RANDOM_STATE = 42
SYSTEM      = Params.SYSTEM
TARGET   = 'Surcharge_Ratio'
YEAR_COL = 'Inspection_Year'
CUT_YEAR = 2018          # train <= CUT_YEAR ; test > CUT_YEAR
SVR_FINAL_N = 10_000
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- feature lists (must match the baseline regression script) ----
MAINTENANCE_FEATURES = sorted(['Blockages_SinceLastInsp','General_Cleaning_SinceLastInsp',
    'Total_Maintenance_Events_SinceLastInsp','Inspection_SinceLastInsp'])
PIPE_FEATURES = sorted(['Downstream_Pipe_Age_Avg','Downstream_Pipe_Diameter_Avg','Upstream_Pipe_Diameter_Avg',
    'Downstream_Pipe_Length_Avg','Upstream_Pipe_Length_Avg','Downstream_Pipe_Slope_Avg','Upstream_Pipe_Slope_Avg'])
TOPOGRAPHIC_FEATURES = sorted(['Elev'])
LANDUSE_FEATURES = sorted(['PopDensity_m2','AADT'])
MANHOLE_CONFIG_FEATURES = sorted(['Num_Inlet_Pipes','Reservoir_Distance'])
WATER_SILT_CLIMATE_FEATURES = sorted(['Silt_Coverage_Ratio','Temp_Mean_7d_Avg','Rain_7d_Sum','Windspeed_Day','Dewpoint_Day'])
CATEGORICAL_FEATURES = sorted(['Geol_Type'])
_seen=set(); ALL_FEATURES=[]
for f in (MAINTENANCE_FEATURES+PIPE_FEATURES+TOPOGRAPHIC_FEATURES+LANDUSE_FEATURES+
          MANHOLE_CONFIG_FEATURES+WATER_SILT_CLIMATE_FEATURES+CATEGORICAL_FEATURES):
    if f not in _seen: ALL_FEATURES.append(f); _seen.add(f)

# ---- helpers (same encoding/scaling as baseline) ----
def safe_label_encode(tr,te):
    ts=tr.astype(str).str.strip(); vs=te.astype(str).str.strip()
    le=LabelEncoder().fit(ts); unseen=~vs.isin(le.classes_)
    if unseen.any(): vs=vs.copy(); vs[unseen]=ts.value_counts().index[0]
    return le.transform(ts), le.transform(vs), le
def encode_cats(Xtr,Xte,cats):
    Xtr,Xte=Xtr.copy(),Xte.copy()
    for col in cats:
        tr_e,_,le=safe_label_encode(Xtr[col],Xtr[col]); Xtr[col]=tr_e
        ts=Xte[col].astype(str).str.strip(); ts[~ts.isin(le.classes_)]=le.classes_[0]; Xte[col]=le.transform(ts)
    return Xtr,Xte
def scale_num(Xtr,Xte,cats):
    num=[c for c in Xtr.columns if c not in cats]; Xtr,Xte=Xtr.copy(),Xte.copy()
    sc=StandardScaler(); Xtr[num]=sc.fit_transform(Xtr[num]); Xte[num]=sc.transform(Xte[num])
    return Xtr,Xte,sc

def load_params(model_name):
    p=os.path.join(BASE_DIR,f"{model_name.replace(' ','_')}_params.json")
    if not os.path.exists(p):
        print(f"  !! no params for {model_name} at {p} — skipping"); return None
    with open(p) as f: d=json.load(f)
    return d['best_params']

def clean_tree_params(bp):
    """Strip leftovers / re-supplied args to avoid duplicate-keyword errors."""
    bp=dict(bp)
    for k in ('cat_features','verbose','thread_count'): bp.pop(k,None)
    return bp

def save_row(name, m):
    sf=os.path.join(OUTPUT_DIR,'holdout_temporal_summary.csv')
    row=pd.DataFrame([{'Model':name,'R²':m['R²'],'MAE':m['MAE'],'RMSE':m['RMSE'],
                       'Split':'temporal_<=2018_vs_2019-2021','Timestamp':datetime.now().isoformat()}])
    (pd.concat([pd.read_csv(sf),row],ignore_index=True) if os.path.exists(sf) else row).to_csv(sf,index=False)

# ---- load + temporal split ----
print("="*70,"\nTEMPORAL HOLD-OUT (regression) — train<=2018, test 2019-2021\n","="*70)
df=pd.read_csv(DATA_FILE,low_memory=False)
df=df[df['System']==SYSTEM].copy()
avail=[f for f in ALL_FEATURES if f in df.columns]; cats=[f for f in CATEGORICAL_FEATURES if f in avail]
X=df[avail].copy(); y=df[TARGET].copy(); yr=df[YEAR_COL]
mask=y.notna(); X,y,yr=X[mask],y[mask],yr[mask]
for col in avail:
    if X[col].isna().any(): X[col]=X[col].fillna('Unknown') if col in cats else X[col].fillna(X[col].median())

tr = yr<=CUT_YEAR; te = yr>CUT_YEAR
X_train,y_train=X[tr].reset_index(drop=True),y[tr].reset_index(drop=True)
X_test, y_test =X[te].reset_index(drop=True),y[te].reset_index(drop=True)
print(f"  Year distribution:\n{yr.value_counts().sort_index().to_string()}")
print(f"\n  Train (<= {CUT_YEAR}): {len(X_train):,} | Test (> {CUT_YEAR}): {len(X_test):,}  "
      f"({len(X_test)/(len(X_train)+len(X_test))*100:.0f}% test)")

models = ['LightGBM','XGBoost','CatBoost','Random Forest','Gradient Boosting','MLP','AdaBoost','SVR']
for name in models:
    bp=load_params(name)
    if bp is None: continue
    t0=datetime.now()
    try:
        if name=='LightGBM':
            Xtr,Xte=X_train.copy(),X_test.copy()
            for c in cats: Xtr[c]=Xtr[c].astype('category'); Xte[c]=Xte[c].astype('category')
            fm=lgb.LGBMRegressor(**clean_tree_params(bp)).fit(Xtr,y_train); yp=np.clip(fm.predict(Xte),0,None)
        elif name=='XGBoost':
            Xtr,Xte=encode_cats(X_train,X_test,cats)
            fm=xgb.XGBRegressor(**clean_tree_params(bp)).fit(Xtr,y_train); yp=np.clip(fm.predict(Xte),0,None)
        elif name=='CatBoost':
            Xtr,Xte=X_train.copy(),X_test.copy()
            for c in cats: Xtr[c]=Xtr[c].astype(str); Xte[c]=Xte[c].astype(str)
            fm=cb.CatBoostRegressor(**{**bp,'cat_features':cats,'verbose':False}).fit(Xtr,y_train,verbose=False); yp=np.clip(fm.predict(Xte),0,None)
        elif name=='Random Forest':
            Xtr,Xte=encode_cats(X_train,X_test,cats)
            fm=RandomForestRegressor(**bp).fit(Xtr,y_train); yp=np.clip(fm.predict(Xte),0,None)
        elif name=='Gradient Boosting':
            Xtr,Xte=encode_cats(X_train,X_test,cats)
            fm=GradientBoostingRegressor(**bp).fit(Xtr,y_train); yp=np.clip(fm.predict(Xte),0,None)
        elif name=='MLP':
            Xtr,Xte=encode_cats(X_train,X_test,cats); Xtr,Xte,_=scale_num(Xtr,Xte,cats)
            mp=dict(bp); 
            if isinstance(mp.get('hidden_layer_sizes'),list): mp['hidden_layer_sizes']=tuple(mp['hidden_layer_sizes'])
            fm=MLPRegressor(**mp).fit(Xtr,y_train); yp=np.clip(fm.predict(Xte),0,None)
        elif name=='AdaBoost':
            Xtr,Xte=encode_cats(X_train,X_test,cats)
            ap=dict(bp)
            # reconstruct base estimator if it was serialised as a repr string
            if 'estimator' in ap and isinstance(ap['estimator'],str):
                import re
                md=re.search(r'max_depth=(\d+)',ap['estimator']); depth=int(md.group(1)) if md else 3
                ap['estimator']=DecisionTreeRegressor(max_depth=depth,random_state=RANDOM_STATE)
            fm=AdaBoostRegressor(**ap).fit(Xtr,y_train); yp=np.clip(fm.predict(Xte),0,None)
        elif name=='SVR':
            Xtr,Xte=encode_cats(X_train,X_test,cats); Xtr,Xte,_=scale_num(Xtr,Xte,cats)
            idx=np.random.default_rng(RANDOM_STATE).choice(len(Xtr),min(SVR_FINAL_N,len(Xtr)),replace=False)
            fm=SVR(**bp).fit(Xtr.iloc[idx],y_train.iloc[idx]); yp=np.clip(fm.predict(Xte),0,None)
        m={'R²':r2_score(y_test,yp),'MAE':mean_absolute_error(y_test,yp),
           'RMSE':float(np.sqrt(mean_squared_error(y_test,yp)))}
        print(f"  {name:<20} R²={m['R²']:.4f}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"({(datetime.now()-t0).total_seconds():.0f}s)")
        save_row(name,m)
    except Exception as e:
        print(f"  {name:<20} ERROR: {e}")

print(f"\nSaved -> {os.path.join(OUTPUT_DIR,'holdout_temporal_summary.csv')}")
