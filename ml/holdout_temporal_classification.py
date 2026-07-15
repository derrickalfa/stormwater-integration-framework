#!/usr/bin/env python3
"""
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
from config import Paths, Params

holdout_temporal_classification.py
==================================
R2-04 (temporal hold-out) for the OCCURRENCE/CLASSIFICATION tier.

Train on inspections <=2018, test on 2019-2021. Robustness check, not a
replacement for the manhole-based split.

Param-reuse mode (loads baseline {Model}_params.json, no Optuna). IMPORTANT
correctness point: class-balance terms (class_weight / scale_pos_weight /
sample_weight) are RECOMPUTED from THIS split's training set, because surcharge
prevalence differs from the baseline split. The searched hyperparameters are
reused; the balance terms are not.

Threshold: kept consistent with the baseline (selected on the test set), so the
hold-out is directly comparable to the baseline. If you later adopt the
validation-set threshold fix, apply it uniformly to baseline + hold-outs.
"""

import pandas as pd, numpy as np, os, json, re
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, average_precision_score, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb, xgboost as xgb, catboost as cb

# ============================ CONFIG ============================
# All paths resolved from config.py at the repository root.
# Edit INPUT_ROOT and OUTPUT_ROOT there — do not edit paths here.
SYSTEM      = Params.SYSTEM   # 'Stormwater'
DATA_FILE    = str(Paths.FINAL_CLASSIFICATION)
OUTPUT_DIR   = str(Paths.ML_CLASSIFICATION_TEMPORAL)
BASE_DIR     = str(Paths.ML_CLASSIFICATION_DIR)

RANDOM_STATE = 42
SYSTEM      = Params.SYSTEM
TARGET   = 'Surcharged_Binary_Original'
YEAR_COL = 'Inspection_Year'
CUT_YEAR = 2018
SVC_FINAL_N = 20_000
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- feature lists (must match the baseline classification script) ----
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
    sc=StandardScaler(); Xtr[num]=sc.fit_transform(Xtr[num]); Xte[num]=sc.transform(Xte[num]); return Xtr,Xte,sc
def load_params(name):
    p=os.path.join(BASE_DIR,f"{name.replace(' ','_')}_params.json")
    if not os.path.exists(p): print(f"  !! no params for {name}"); return None
    with open(p) as f: return json.load(f)['best_params']
def find_threshold(y_true,yp):
    pr,rc,th=precision_recall_curve(y_true,yp)
    f1=2*pr*rc/(pr+rc+1e-12)
    i=int(np.nanargmax(f1[:-1])) if len(th)>0 else 0
    return th[i] if len(th)>0 else 0.5

def strip_balance(bp):
    """Remove baked-in balance terms AND args we re-supply explicitly
    (random_state/n_jobs/verbose/eval_metric), to avoid duplicate-keyword errors."""
    bp=dict(bp)
    for k in ('class_weight','scale_pos_weight','auto_class_weights','cat_features',
              'random_state','n_jobs','verbose','eval_metric','probability','thread_count'):
        bp.pop(k,None)
    return bp

print("="*70,"\nTEMPORAL HOLD-OUT (classification) — train<=2018, test 2019-2021\n","="*70)
df=pd.read_csv(DATA_FILE,low_memory=False); df=df[df['System']==SYSTEM].copy()
avail=[f for f in ALL_FEATURES if f in df.columns]; cats=[f for f in CATEGORICAL_FEATURES if f in avail]
X=df[avail].copy(); y=df[TARGET].copy(); yr=df[YEAR_COL]
mask=y.notna(); X,y,yr=X[mask],y[mask],yr[mask]
for col in avail:
    if X[col].isna().any(): X[col]=X[col].fillna('Unknown') if col in cats else X[col].fillna(X[col].median())

tr=yr<=CUT_YEAR; te=yr>CUT_YEAR
X_train,y_train=X[tr].reset_index(drop=True),y[tr].reset_index(drop=True)
X_test, y_test =X[te].reset_index(drop=True),y[te].reset_index(drop=True)
print(f"  Train (<= {CUT_YEAR}): {len(X_train):,}  (pos={int(y_train.sum())}, prev={y_train.mean()*100:.2f}%)")
print(f"  Test  (>  {CUT_YEAR}): {len(X_test):,}  (pos={int(y_test.sum())}, prev={y_test.mean()*100:.2f}%)")

# Recompute class-balance terms from THIS training set
cw_arr=compute_class_weight('balanced',classes=np.array([0,1]),y=y_train)
cw_dict={0:float(cw_arr[0]),1:float(cw_arr[1])}; spw=cw_dict[1]/cw_dict[0]
sw_train=compute_sample_weight('balanced',y_train)
print(f"  Recomputed class weight ratio: {spw:.2f}:1")

def evaluate(name,yp):
    thr=find_threshold(y_test,yp); pred=(yp>=thr).astype(int)
    return {'F1':f1_score(y_test,pred),'Recall':recall_score(y_test,pred),
            'Precision':precision_score(y_test,pred),'ROC_AUC':roc_auc_score(y_test,yp),
            'PR_AUC':average_precision_score(y_test,yp),'Threshold':float(thr)}

models=['LightGBM','XGBoost','CatBoost','Random Forest','Gradient Boosting','MLP','AdaBoost','SVC']
for name in models:
    bp=load_params(name)
    if bp is None: continue
    t0=datetime.now()
    try:
        if name=='LightGBM':
            Xtr,Xte=X_train.copy(),X_test.copy()
            for c in cats: Xtr[c]=Xtr[c].astype('category'); Xte[c]=Xte[c].astype('category')
            fm=lgb.LGBMClassifier(**strip_balance(bp),class_weight=cw_dict,random_state=RANDOM_STATE,n_jobs=-1,verbose=-1).fit(Xtr,y_train)
            yp=fm.predict_proba(Xte)[:,1]
        elif name=='XGBoost':
            Xtr,Xte=encode_cats(X_train,X_test,cats)
            fm=xgb.XGBClassifier(**strip_balance(bp),scale_pos_weight=spw,random_state=RANDOM_STATE,n_jobs=-1,eval_metric='logloss').fit(Xtr,y_train)
            yp=fm.predict_proba(Xte)[:,1]
        elif name=='CatBoost':
            Xtr,Xte=X_train.copy(),X_test.copy()
            for c in cats: Xtr[c]=Xtr[c].astype(str); Xte[c]=Xte[c].astype(str)
            fm=cb.CatBoostClassifier(**strip_balance(bp),cat_features=cats,auto_class_weights='Balanced',random_state=RANDOM_STATE,verbose=False).fit(Xtr,y_train,verbose=False)
            yp=fm.predict_proba(Xte)[:,1]
        elif name=='Random Forest':
            Xtr,Xte=encode_cats(X_train,X_test,cats)
            fm=RandomForestClassifier(**strip_balance(bp),class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-1).fit(Xtr,y_train)
            yp=fm.predict_proba(Xte)[:,1]
        elif name=='Gradient Boosting':
            Xtr,Xte=encode_cats(X_train,X_test,cats)
            fm=GradientBoostingClassifier(**strip_balance(bp),random_state=RANDOM_STATE).fit(Xtr,y_train,sample_weight=sw_train)
            yp=fm.predict_proba(Xte)[:,1]
        elif name=='MLP':
            Xtr,Xte=encode_cats(X_train,X_test,cats); Xtr,Xte,_=scale_num(Xtr,Xte,cats); mp=strip_balance(bp)
            if isinstance(mp.get('hidden_layer_sizes'),list): mp['hidden_layer_sizes']=tuple(mp['hidden_layer_sizes'])
            fm=MLPClassifier(**mp,random_state=RANDOM_STATE).fit(Xtr,y_train)  # MLP has no sample_weight
            yp=fm.predict_proba(Xte)[:,1]
        elif name=='AdaBoost':
            Xtr,Xte=encode_cats(X_train,X_test,cats); ap=strip_balance(bp)
            if 'estimator' in ap and isinstance(ap['estimator'],str):
                md=re.search(r'max_depth=(\d+)',ap['estimator']); ap['estimator']=DecisionTreeClassifier(max_depth=int(md.group(1)) if md else 3,random_state=RANDOM_STATE)
            fm=AdaBoostClassifier(**ap,random_state=RANDOM_STATE).fit(Xtr,y_train,sample_weight=sw_train)
            yp=fm.predict_proba(Xte)[:,1]
        elif name=='SVC':
            Xtr,Xte=encode_cats(X_train,X_test,cats); Xtr,Xte,_=scale_num(Xtr,Xte,cats)
            idx=np.random.default_rng(RANDOM_STATE).choice(len(Xtr),min(SVC_FINAL_N,len(Xtr)),replace=False)
            fm=SVC(**strip_balance(bp),class_weight='balanced',probability=True,random_state=RANDOM_STATE).fit(Xtr.iloc[idx],y_train.iloc[idx])
            yp=fm.predict_proba(Xte)[:,1]
        m=evaluate(name,yp)
        print(f"  {name:<20} F1={m['F1']:.4f} Rec={m['Recall']:.4f} Prec={m['Precision']:.4f} "
              f"ROC={m['ROC_AUC']:.4f} PR={m['PR_AUC']:.4f}  ({(datetime.now()-t0).total_seconds():.0f}s)")
        sf=os.path.join(OUTPUT_DIR,'holdout_temporal_summary.csv')
        row=pd.DataFrame([{'Model':name,**m,'Split':'temporal_<=2018_vs_2019-2021'}])
        (pd.concat([pd.read_csv(sf),row],ignore_index=True) if os.path.exists(sf) else row).to_csv(sf,index=False)
    except Exception as e:
        print(f"  {name:<20} ERROR: {e}")

print(f"\nSaved -> {os.path.join(OUTPUT_DIR,'holdout_temporal_summary.csv')}")
