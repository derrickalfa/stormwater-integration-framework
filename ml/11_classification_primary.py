# -*- coding: utf-8 -*-
"""
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)
from config import Paths, Params

================================================================================
MANHOLE-BASED SPLIT CLASSIFICATION — 8 MODELS + FULL SHAP FOR TOP-2
================================================================================
FIXES APPLIED:
  1. AdaBoost: removed class_weight='balanced' from DecisionTreeClassifier
     (was double-weighting alongside sample_weight → F1=0.09, Recall=0.91)
  2. classification_summary.csv: deduplicated before top-2 SHAP selection
  3. make_json_serializable: already present, handles DecisionTreeClassifier JSON crash
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import warnings, os, pickle, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, roc_curve,
                             average_precision_score, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

try:
    import shap;  print(f"✓ SHAP {shap.__version__}")
except ImportError:
    shap = None;  print("⚠ SHAP not installed — pip install shap")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_FILE    = str(Paths.FINAL_CLASSIFICATION)
OUTPUT_DIR   = str(Paths.ML_CLASSIFICATION_DIR)

RANDOM_STATE  = 42
SYSTEM        = 'Stormwater'
TARGET        = 'Surcharged_Binary_Original'
VAL_FRAC      = 0.15
TEST_FRAC     = 0.20

N_FAST   = 50
N_MEDIUM = 30
N_SLOW   = 15
N_SVC    = 15

SVC_TRIAL_N = 8_000
SVC_FINAL_N = 20_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'training_checkpoint.json')

# ==============================================================================
# FEATURE LIST — FIXED SORTED ORDER
# ==============================================================================

MAINTENANCE_FEATURES = sorted([
    # 'Blockages_SinceLastInsp', 'General_Cleaning_SinceLastInsp',
    # 'Total_Maintenance_Events_SinceLastInsp', 'Inspection_SinceLastInsp'
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

print(f"\n📋 Features: {len(ALL_FEATURES)}  (fixed order, no interactions)")

# ==============================================================================
# HELPERS
# ==============================================================================

def safe_label_encode(tr, te):
    ts = tr.astype(str).str.strip(); vs = te.astype(str).str.strip()
    le = LabelEncoder().fit(ts)
    unseen = ~vs.isin(le.classes_)
    if unseen.any():
        vs = vs.copy(); vs[unseen] = ts.value_counts().index[0]
    return le.transform(ts), le.transform(vs), le

def encode_cats_numeric(Xtr, Xva, Xte, cat_cols):
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for col in cat_cols:
        tr_e, va_e, le = safe_label_encode(Xtr[col], Xva[col])
        Xtr[col] = tr_e; Xva[col] = va_e
        ts = Xte[col].astype(str).str.strip()
        ts[~ts.isin(le.classes_)] = le.classes_[0]
        Xte[col] = le.transform(ts)
    return Xtr, Xva, Xte

def encode_cats_category(Xtr, Xva, Xte, cat_cols):
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    for col in cat_cols:
        for d in [Xtr, Xva, Xte]: d[col] = d[col].astype('category')
    return Xtr, Xva, Xte

def scale_numeric(Xtr, Xva, Xte, cat_cols):
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    Xtr, Xva, Xte = Xtr.copy(), Xva.copy(), Xte.copy()
    sc = StandardScaler()
    Xtr[num_cols] = sc.fit_transform(Xtr[num_cols])
    Xva[num_cols] = sc.transform(Xva[num_cols])
    Xte[num_cols] = sc.transform(Xte[num_cols])
    return Xtr, Xva, Xte, sc

def subsample_rows(X, y, n, sw=None, seed=RANDOM_STATE):
    if n is None or len(X) <= n:
        return (X.reset_index(drop=True), y.reset_index(drop=True),
                sw if sw is None else sw[np.arange(len(X))])
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return (X.iloc[idx].reset_index(drop=True),
            y.iloc[idx].reset_index(drop=True),
            None if sw is None else sw[idx])

def find_optimal_threshold(y_true, y_proba):
    precs, recs, thrs = precision_recall_curve(y_true, y_proba)
    f1s = 2*(precs*recs)/(precs+recs+1e-10)
    idx = np.argmax(f1s)
    return (thrs[idx] if idx < len(thrs) else 0.5), f1s[idx]

def compute_metrics(y_true, y_pred, y_proba, t_elapsed):
    return {
        'Accuracy':      float(accuracy_score(y_true, y_pred)),
        'Precision':     float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':        float(recall_score(y_true, y_pred, zero_division=0)),
        'F1':            float(f1_score(y_true, y_pred, zero_division=0)),
        'ROC_AUC':       float(roc_auc_score(y_true, y_proba)),
        'Avg_Precision': float(average_precision_score(y_true, y_proba)),
        'Brier':         float(brier_score_loss(y_true, y_proba)),
        'Combined':      float(0.5*f1_score(y_true,y_pred,zero_division=0) +
                               0.3*recall_score(y_true,y_pred,zero_division=0) +
                               0.2*roc_auc_score(y_true, y_proba)),
        'training_time': t_elapsed
    }

def objective_score(y_true, y_proba):
    t, _ = find_optimal_threshold(y_true, y_proba)
    yp   = (y_proba >= t).astype(int)
    f1   = f1_score(y_true, yp, zero_division=0)
    rec  = recall_score(y_true, yp, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)
    return 0.5*f1 + 0.3*rec + 0.2*auc, f1, rec, auc

def make_json_serializable(obj):
    """Recursively convert to JSON-safe types. Sklearn objects → repr string."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)  # catches DecisionTreeClassifier etc.

def save_results(name, model, params, metrics, threshold, history):
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    pkl = os.path.join(OUTPUT_DIR, f'{name.replace(" ","_")}_{ts}.pkl')
    with open(pkl, 'wb') as f:
        pickle.dump({'model': model, 'model_name': name, 'best_params': params,
                     'metrics': metrics, 'optimal_threshold': threshold,
                     'timestamp': ts}, f)
    with open(os.path.join(OUTPUT_DIR, f'{name.replace(" ","_")}_params.json'), 'w') as f:
        json.dump({'model_name': name,
                   'best_params': make_json_serializable(params),
                   'metrics':     make_json_serializable(metrics),
                   'optimal_threshold': float(threshold),
                   'timestamp': ts}, f, indent=2)
    if history:
        pd.DataFrame(history).to_csv(
            os.path.join(OUTPUT_DIR, f'{name.replace(" ","_")}_trials.csv'), index=False)
    sf  = os.path.join(OUTPUT_DIR, 'classification_summary.csv')
    row = pd.DataFrame([{'Model': name, 'F1': metrics['F1'], 'Recall': metrics['Recall'],
                          'Precision': metrics['Precision'], 'ROC_AUC': metrics['ROC_AUC'],
                          'Avg_Precision': metrics['Avg_Precision'], 'Brier': metrics['Brier'],
                          'Combined': metrics['Combined'], 'Threshold': threshold,
                          'Training_Time_s': metrics['training_time'], 'Timestamp': ts}])
    (pd.concat([pd.read_csv(sf), row], ignore_index=True) if os.path.exists(sf) else row
     ).to_csv(sf, index=False)
    print(f"  ✓ Saved: {name}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f: return json.load(f)
    return {'completed_models': []}

def save_checkpoint(done):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'completed_models': done}, f)

def print_metrics(name, y_test, y_pred, y_proba, m, thresh):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Pred 0        Pred 1")
    print(f"  Actual 0     {tn:>10,}  {fp:>10,}")
    print(f"  Actual 1     {fn:>10,}  {tp:>10,}")
    print(f"\n  ROC-AUC   : {m['ROC_AUC']:.4f}")
    print(f"  Avg Prec  : {m['Avg_Precision']:.4f}")
    print(f"  Brier     : {m['Brier']:.4f}")
    print(f"  F1        : {m['F1']:.4f}")
    print(f"  Recall    : {m['Recall']:.4f}  ({m['Recall']*100:.1f}% of {tp+fn} surcharges caught)")
    print(f"  Precision : {m['Precision']:.4f}")
    print(f"  Combined  : {m['Combined']:.4f}  (0.5F1+0.3Rec+0.2AUC)")
    print(f"  Threshold : {thresh:.3f}")

# ==============================================================================
# DIAGNOSTIC PLOTS
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, name, out):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = np.array([[f'TN\n{tn:,}', f'FP\n{fp:,}'], [f'FN\n{fn:,}', f'TP\n{tp:,}']])
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=axes[0],
                xticklabels=['Pred: No Surge', 'Pred: Surge'],
                yticklabels=['Act: No Surge', 'Act: Surge'],
                linewidths=2, linecolor='white', cbar=False)
    axes[0].set_title(f'Confusion Matrix — {name}\n(raw counts)', fontweight='bold')
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels_p = np.array([[f'TN\n{cm_n[0,0]:.1%}', f'FP\n{cm_n[0,1]:.1%}'],
                          [f'FN\n{cm_n[1,0]:.1%}', f'TP\n{cm_n[1,1]:.1%}']])
    sns.heatmap(cm_n, annot=labels_p, fmt='', cmap='RdYlGn', ax=axes[1],
                xticklabels=['Pred: No Surge', 'Pred: Surge'],
                yticklabels=['Act: No Surge', 'Act: Surge'],
                vmin=0, vmax=1, linewidths=2, linecolor='white', cbar=True)
    axes[1].set_title(f'Confusion Matrix — {name}\n(row-normalised %)', fontweight='bold')
    plt.suptitle(f'{name} — Recall={tp/(tp+fn):.3f}  Precision={tp/(tp+fp):.3f}  '
                 f'Flagged={tp+fp:,}  Missed={fn:,}', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close(); print(f"  ✓ confusion_matrix.png")

def plot_threshold_analysis(y_true, y_proba, thresh, name, out):
    thrs = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs = [], [], []
    for t in thrs:
        p = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, p, zero_division=0))
        precs.append(precision_score(y_true, p, zero_division=0))
        recs.append(recall_score(y_true, p, zero_division=0))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(thrs, f1s, color='steelblue', lw=2.5, label='F1')
    axes[0].plot(thrs, precs, color='darkorange', lw=2, ls='--', label='Precision')
    axes[0].plot(thrs, recs, color='tomato', lw=2, ls=':', label='Recall')
    axes[0].axvline(thresh, color='black', ls='--', lw=2, label=f'Threshold ({thresh:.3f})')
    axes[0].set_title(f'{name} — Threshold Sensitivity', fontweight='bold')
    axes[0].set_xlabel('Threshold'); axes[0].legend(); axes[0].grid(alpha=0.3)
    pos = y_proba[y_true==1]; neg = y_proba[y_true==0]
    axes[1].hist(neg, bins=60, alpha=0.6, color='steelblue', density=True,
                 label=f'No Surcharge (n={len(neg):,})')
    axes[1].hist(pos, bins=60, alpha=0.7, color='tomato', density=True,
                 label=f'Surcharge (n={len(pos):,})')
    axes[1].axvline(thresh, color='black', ls='--', lw=2, label=f'Threshold ({thresh:.3f})')
    axes[1].set_title(f'{name} — Score Distribution', fontweight='bold')
    axes[1].set_xlabel('Predicted Probability'); axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'threshold_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close(); print(f"  ✓ threshold_analysis.png")

def plot_roc_pr_curves(models_data, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'purple',
              '#1abc9c', '#e67e22', '#9b59b6']
    for i, (name, yt, yp) in enumerate(models_data):
        c = colors[i % len(colors)]
        fpr, tpr, _ = roc_curve(yt, yp)
        axes[0].plot(fpr, tpr, color=c, lw=2,
                     label=f'{name}  AUC={roc_auc_score(yt,yp):.4f}')
        prec, rec, _ = precision_recall_curve(yt, yp)
        axes[1].plot(rec, prec, color=c, lw=2,
                     label=f'{name}  AP={average_precision_score(yt,yp):.4f}')
    axes[0].plot([0,1], [0,1], 'k--', alpha=0.4, label='Random')
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    base = np.mean(models_data[0][1])
    axes[1].axhline(base, color='k', ls='--', alpha=0.4, label=f'Random (P={base:.3f})')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.suptitle('All Models — ROC & Precision-Recall', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'roc_pr_curves.png'), dpi=200, bbox_inches='tight')
    plt.close(); print(f"  ✓ roc_pr_curves.png")

def plot_calibration(models_data, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'purple',
              '#1abc9c', '#e67e22', '#9b59b6']
    for i, (name, yt, yp) in enumerate(models_data):
        fp, mp = calibration_curve(yt, yp, n_bins=15, strategy='quantile')
        ax.plot(mp, fp, 's-', color=colors[i], lw=2, markersize=6,
                label=f'{name}  Brier={brier_score_loss(yt,yp):.4f}')
    ax.plot([0,1], [0,1], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction Positives')
    ax.set_title('Calibration Curves', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'calibration_curve.png'), dpi=200, bbox_inches='tight')
    plt.close(); print(f"  ✓ calibration_curve.png")

def plot_model_comparison(sdf, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, col, title in [
        (axes[0], 'F1',      'F1 Score'),
        (axes[1], 'ROC_AUC', 'ROC-AUC'),
        (axes[2], 'Recall',  'Recall'),
    ]:
        sub  = sdf.sort_values(col, ascending=False).reset_index(drop=True)
        cols = ['gold' if i==0 else '#3498db' for i in range(len(sub))]
        ax.barh(range(len(sub)), sub[col], color=cols, alpha=0.85)
        ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub['Model'], fontsize=9)
        ax.invert_yaxis(); ax.set_title(title, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(row[col]+0.002, i, f'{row[col]:.4f}', va='center', fontsize=8)
    plt.suptitle('8-Model Classification Comparison — Manhole-Based Split',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'model_comparison_8models.png'), dpi=200, bbox_inches='tight')
    plt.close(); print(f"  ✓ model_comparison_8models.png")

# ==============================================================================
# FULL SHAP ANALYSIS
# ==============================================================================

CATEGORY_COLORS = {
    'Maintenance/Defects': '#e74c3c', 'Climate/Weather':  '#3498db',
    'Pipe Properties':     '#2ecc71', 'Topographic':       '#9b59b6',
    'Land Use / Urban':    '#f39c12', 'Manhole Config':    '#1abc9c',
    'Silt/Capacity':       '#e67e22', 'Other':             '#95a5a6',
}

def classify_feature(f):
    fl = f.lower()
    if any(k in fl for k in ['silt','capacity']):                        return 'Silt/Capacity'
    if any(k in fl for k in ['rain','temp','humid','wind','dewpoint']):   return 'Climate/Weather'
    if any(k in fl for k in ['pipe','upstream','downstream','slope',
                               'diameter','length','age_avg']):           return 'Pipe Properties'
    if any(k in fl for k in ['elev','curvat']):                          return 'Topographic'
    if any(k in fl for k in ['bldg','pop','catch','imperv','aadt',
                               'landuse']):                               return 'Land Use / Urban'
    if any(k in fl for k in ['num_inlet','num_upstream','num_downstream',
                               'reservoir','prop_manhole','inspection_year',
                               'inspection_month','age_at','size_of_cover',
                               'type','geol','road']):                    return 'Manhole Config'
    if any(k in fl for k in ['clean','maintenance','crack','defect','deform',
                               'infiltr','corros','environ','inspect','broken',
                               'connect','flow','block','general']):      return 'Maintenance/Defects'
    return 'Other'

def run_full_shap_analysis(model, model_name, X_te, y_te, y_proba,
                            threshold, metrics, cat_cols, X_tr=None, is_tree=True):
    out = os.path.join(OUTPUT_DIR, f'SHAP_{model_name.replace(" ","_")}')
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FULL SHAP — {model_name}  ({'TreeExplainer' if is_tree else 'KernelExplainer'})")
    print(f"{'='*70}")

    X_shap = X_te.copy()
    if model_name != 'LightGBM' and model_name != 'CatBoost':
        for col in cat_cols:
            if col in X_shap.columns:
                if X_shap[col].dtype.name == 'category':
                    X_shap[col] = X_shap[col].cat.codes.astype('float32')
                else:
                    X_shap[col] = LabelEncoder().fit_transform(
                        X_shap[col].astype(str)).astype('float32')
        for col in X_shap.select_dtypes(include=[np.number]).columns:
            X_shap[col] = X_shap[col].astype('float32')

    y_pred = (y_proba >= threshold).astype(int)
    y_arr  = np.array(y_te)
    tp_idx = np.where((y_pred==1)&(y_arr==1))[0]
    fp_idx = np.where((y_pred==1)&(y_arr==0))[0]
    fn_idx = np.where((y_pred==0)&(y_arr==1))[0]
    tn_idx = np.where((y_pred==0)&(y_arr==0))[0]
    np.random.seed(RANDOM_STATE)
    sample_idx = np.concatenate([
        np.random.choice(tp_idx, min(50,len(tp_idx)),  replace=False),
        np.random.choice(fp_idx, min(50,len(fp_idx)),  replace=False),
        np.random.choice(fn_idx, min(50,len(fn_idx)),  replace=False),
        np.random.choice(tn_idx, min(350,len(tn_idx)), replace=False),
    ])
    np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:min(500, len(sample_idx))]

    X_s    = X_shap.iloc[sample_idx].copy()
    y_s    = y_arr[sample_idx]
    p_s    = y_proba[sample_idx]

    X_plot = X_s.copy()
    for col in cat_cols:
        if col in X_plot.columns and X_plot[col].dtype.name == 'category':
            X_plot[col] = X_plot[col].cat.codes.astype('float32')

    feat_names = X_plot.columns.tolist()

    print(f"  Computing SHAP ({len(sample_idx)} stratified samples) …")
    t0 = datetime.now()

    if is_tree:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_s)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values)==2 else shap_values[0]
        ev = explainer.expected_value
        if hasattr(ev, '__len__'):
            base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            base_val = float(ev)
    else:
        KBGR = 300; KEXP = 500
        X_bg = shap.sample(X_tr if X_tr is not None else X_s, KBGR,
                            random_state=RANDOM_STATE)
        print(f"  KernelExplainer: bg={KBGR}, explain={KEXP} rows …")
        explainer   = shap.KernelExplainer(lambda x: model.predict_proba(x)[:,1], X_bg)
        shap_values = np.array(explainer.shap_values(X_s.iloc[:KEXP]))
        X_s    = X_s.iloc[:KEXP].copy()
        X_plot = X_plot.iloc[:KEXP].copy()
        y_s    = y_s[:KEXP]; p_s = p_s[:KEXP]
        base_val   = float(explainer.expected_value)
        feat_names = X_plot.columns.tolist()

    print(f"  ✓ SHAP done in {(datetime.now()-t0).total_seconds():.1f}s")

    imp_df = pd.DataFrame({
        'Feature':   feat_names,
        'SHAP_Mean': np.abs(shap_values).mean(axis=0),
        'Category':  [classify_feature(f) for f in feat_names]
    }).sort_values('SHAP_Mean', ascending=False).reset_index(drop=True)
    imp_df['SHAP_Pct']   = imp_df['SHAP_Mean'] / imp_df['SHAP_Mean'].sum() * 100
    imp_df['Cumulative'] = imp_df['SHAP_Pct'].cumsum()
    imp_df.to_csv(os.path.join(out, 'feature_importance.csv'), index=False)

    top6  = imp_df['Feature'].head(6).tolist()
    top10 = imp_df['Feature'].head(10).tolist()
    top25 = imp_df['Feature'].head(25).tolist()

    # 1. Summary Bar
    top40  = imp_df.head(40)
    colors = [CATEGORY_COLORS.get(c, '#95a5a6') for c in top40['Category']]
    fig, ax = plt.subplots(figsize=(12, max(8, len(top40)*0.32)))
    ax.barh(range(len(top40)), top40['SHAP_Mean'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top40))); ax.set_yticklabels(top40['Feature'], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel('Mean |SHAP|', fontweight='bold')
    ax.set_title(f'{model_name} — SHAP Importance (Top 40)\n'
                 f'F1={metrics["F1"]:.4f}  AUC={metrics["ROC_AUC"]:.4f}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    mx = top40['SHAP_Mean'].max()
    for i, (v, p) in enumerate(zip(top40['SHAP_Mean'], top40['SHAP_Pct'])):
        ax.text(v+mx*0.005, i, f'{p:.1f}%', va='center', fontsize=7)
    patches = [mpatches.Patch(color=v, label=k) for k,v in CATEGORY_COLORS.items()
               if k in top40['Category'].values]
    ax.legend(handles=patches, loc='lower right', fontsize=7, title='Category')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'summary_bar.png'), dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ summary_bar.png")

    # 2. Beeswarm
    t25_idx = [feat_names.index(f) for f in top25]
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values[:,t25_idx], X_plot.iloc[:,t25_idx],
                      feature_names=top25, show=False, max_display=25)
    plt.title(f'{model_name} — SHAP Beeswarm (Top 25)\n'
              'Red=high value  Blue=low  Right=→P(surcharge)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'beeswarm.png'), dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ beeswarm.png")

    # 3. Waterfall — TP / FN / FP
    yp_s = (p_s >= threshold).astype(int)
    tp_s = np.where((yp_s==1)&(y_s==1))[0]
    fn_s = np.where((yp_s==0)&(y_s==1))[0]
    fp_s = np.where((yp_s==1)&(y_s==0))[0]
    if len(tp_s): tp_s = tp_s[np.argsort(p_s[tp_s])[::-1]]
    if len(fn_s): fn_s = fn_s[np.argsort(p_s[fn_s])[::-1]]
    if len(fp_s): fp_s = fp_s[np.argsort(p_s[fp_s])[::-1]]

    for grp, idxs, title_pfx in [
        ('TP_caught',   tp_s[:3], 'CAUGHT — True Positive'),
        ('FN_missed',   fn_s[:3], 'MISSED — False Negative'),
        ('FP_falsealm', fp_s[:3], 'FALSE ALARM — False Positive'),
    ]:
        if len(idxs) == 0: continue
        n_c = len(idxs)
        fig, axes = plt.subplots(1, n_c, figsize=(7*n_c, 6), squeeze=False)
        for j, idx in enumerate(idxs):
            sv    = shap_values[idx]
            fv    = X_plot.iloc[idx]
            order = np.argsort(np.abs(sv))[::-1][:12]
            sv_t  = sv[order]; fn_t = [feat_names[o] for o in order]
            fv_t  = [fv.iloc[o] for o in order]
            cols  = ['tomato' if v>0 else 'steelblue' for v in sv_t]
            ax    = axes[0][j]
            ax.barh(range(len(sv_t)), sv_t, color=cols, alpha=0.85, edgecolor='white')
            ax.set_yticks(range(len(sv_t)))
            ax.set_yticklabels([f'{fn_t[k][:28]}\n={fv_t[k]:.3g}' if isinstance(fv_t[k], (int, float)) 
                    else f'{fn_t[k][:28]}\n={str(fv_t[k])[:10]}' 
                    for k in range(len(fn_t))], fontsize=8)
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel('SHAP  (→ increases P(surcharge))', fontsize=9)
            ax.set_title(f'Case {j+1}  P={p_s[idx]:.3f}  Actual={"Surge" if y_s[idx]==1 else "No"}',
                         fontsize=9, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        fig.suptitle(f'{model_name} — {title_pfx}\nRed=toward surcharge  Blue=away',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'waterfall_{grp}.png'), dpi=200, bbox_inches='tight')
        plt.close()
    print("  ✓ waterfall_TP.png | waterfall_FN.png | waterfall_FP.png")

    # 4. Dependence plots (top 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, feat in zip(axes.flat, top6):
        fi   = feat_names.index(feat)
        fv   = X_plot[feat].values
        sv_f = shap_values[:,fi]
        best_corr, best_col = 0, None
        for cf in top25:
            if cf == feat: continue
            ci = feat_names.index(cf)
            col_vals = X_plot.iloc[:, ci]
            if not pd.api.types.is_numeric_dtype(col_vals): continue
            r = abs(np.corrcoef(col_vals.values.astype(float), sv_f)[0, 1])
            if np.isfinite(r) and r > best_corr: best_corr, best_col = r, cf
        if best_col and best_corr > 0.05:
            ci = feat_names.index(best_col)
            sc = ax.scatter(fv, sv_f, c=X_plot.iloc[:,ci].values,
                            cmap='coolwarm', alpha=0.5, s=12)
            plt.colorbar(sc, ax=ax, label=f'{best_col[:18]} (r={best_corr:.2f})')
        else:
            ax.scatter(fv, sv_f, alpha=0.4, s=12, color='steelblue')
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel(feat[:30], fontsize=8); ax.set_ylabel('SHAP', fontsize=8)
        ax.set_title(feat[:35], fontweight='bold', fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle(f'{model_name} — SHAP Dependence (Top 6)\nColour=most correlated interacting feature',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'dependence_plots.png'), dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ dependence_plots.png")

    # 5. Interaction heatmap (tree only)
    if is_tree and lgb is not None:
        try:
            t10_idx = [feat_names.index(f) for f in top10]
            X_i     = X_s.iloc[:,t10_idx].copy()
            for col in cat_cols:
                if col in X_i.columns and X_i[col].dtype.name == 'category':
                    X_i[col] = X_i[col].cat.codes.astype('float32')
            m10 = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=6,
                                      random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                                      class_weight='balanced')
            m10.fit(X_i, y_s)
            exp10  = shap.TreeExplainer(m10)
            sv_ix  = exp10.shap_interaction_values(X_i.iloc[:min(300,len(X_i))])
            if isinstance(sv_ix, list): sv_ix = sv_ix[1] if len(sv_ix)==2 else sv_ix[0]
            mat    = np.abs(sv_ix).mean(axis=0).copy(); np.fill_diagonal(mat, 0)

            fig, axes5 = plt.subplots(1, 2, figsize=(18, 7))
            im = axes5[0].imshow(mat, cmap='Reds', aspect='auto')
            axes5[0].set_xticks(range(10)); axes5[0].set_yticks(range(10))
            axes5[0].set_xticklabels([f[:18] for f in top10], rotation=45, ha='right', fontsize=8)
            axes5[0].set_yticklabels([f[:18] for f in top10], fontsize=8)
            axes5[0].set_title(f'{model_name} — SHAP Interaction Matrix', fontweight='bold')
            plt.colorbar(im, ax=axes5[0], label='Mean |Interaction|')
            vmax = mat.max()
            for i2 in range(10):
                for j2 in range(10):
                    if i2 != j2 and mat[i2,j2] > 0:
                        axes5[0].text(j2, i2, f'{mat[i2,j2]:.3f}', ha='center', va='center',
                                      fontsize=6.5,
                                      color='white' if mat[i2,j2]>vmax*0.55 else 'black')
            pairs = [(top10[ii], top10[jj], mat[ii,jj])
                     for ii in range(10) for jj in range(10) if jj > ii]
            pairs_df = (pd.DataFrame(pairs, columns=['F1','F2','Strength'])
                        .sort_values('Strength', ascending=False))
            pairs_df['Pair'] = pairs_df['F1'].str[:20] + '\n× ' + pairs_df['F2'].str[:20]
            axes5[1].barh(pairs_df.head(10)['Pair'], pairs_df.head(10)['Strength'],
                          color='tomato', alpha=0.85)
            axes5[1].invert_yaxis()
            axes5[1].set_title(f'{model_name} — Top 10 Feature Interactions', fontweight='bold')
            axes5[1].set_xlabel('Mean |SHAP Interaction|')
            axes5[1].grid(axis='x', alpha=0.3)
            pairs_df.to_csv(os.path.join(out, 'interactions.csv'), index=False)
            plt.suptitle(f'{model_name} — SHAP Interaction Analysis', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(out, 'interactions.png'), dpi=200, bbox_inches='tight')
            plt.close(); print("  ✓ interactions.png | interactions.csv")
        except Exception as e:
            print(f"  ⚠ Interactions skipped: {e}")
    else:
        print("  ℹ Interactions skipped (non-tree or LightGBM not available)")

    # 6. Category summary
    cat_s = (imp_df.groupby('Category')['SHAP_Pct'].sum()
             .reset_index().sort_values('SHAP_Pct', ascending=False))
    wcols = [CATEGORY_COLORS.get(c, '#95a5a6') for c in cat_s['Category']]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.pie(cat_s['SHAP_Pct'],
            labels=[f"{r['Category']}\n({r['SHAP_Pct']:.1f}%)" for _, r in cat_s.iterrows()],
            colors=wcols, startangle=140)
    ax1.set_title(f'{model_name} — SHAP by Category', fontweight='bold')
    top20 = imp_df.head(20).copy()
    top20['Color'] = top20['Category'].map(CATEGORY_COLORS).fillna('#95a5a6')
    ax2.barh(top20['Feature'], top20['SHAP_Mean'], color=top20['Color'], alpha=0.85)
    ax2.invert_yaxis()
    ax2.set_title('Top 20 Features — coloured by category', fontweight='bold')
    ax2.set_xlabel('Mean |SHAP|'); ax2.grid(axis='x', alpha=0.3)
    patches2 = [mpatches.Patch(color=v, label=k) for k,v in CATEGORY_COLORS.items()
                if k in top20['Category'].values]
    ax2.legend(handles=patches2, fontsize=7, loc='lower right')
    plt.suptitle(f'{model_name} — What Drives Surcharge Predictions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'category_summary.png'), dpi=200, bbox_inches='tight')
    plt.close(); print("  ✓ category_summary.png")

    print(f"\n  Top-15 SHAP features for {model_name}:")
    print(f"  {'Rank':<5} {'Feature':<42} {'%':>6}  {'Cumul':>7}")
    print("  " + "─"*58)
    for i, row in imp_df.head(15).iterrows():
        print(f"  {i+1:<5} {row['Feature']:<42} {row['SHAP_Pct']:>5.1f}%  {row['Cumulative']:>6.1f}%")
    n80 = int((imp_df['Cumulative'] <= 80).sum()) + 1
    print(f"\n  → {n80} features explain 80% of predictions")
    print(f"  ✓ All outputs → {out}")

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n"+"="*70); print("LOADING DATA"); print("="*70)

df_all = pd.read_csv(DATA_FILE, low_memory=False)
df     = df_all[df_all['System']==SYSTEM].copy()
print(f"✓ {SYSTEM}: {len(df):,} records")

avail = [f for f in ALL_FEATURES if f in df.columns]
cats  = [f for f in CATEGORICAL_FEATURES if f in avail]

X = df[avail].copy(); y = df[TARGET].copy()
mask = y.notna(); X, y, df = X[mask], y[mask], df[mask]
for col in avail:
    if X[col].isna().any():
        X[col] = X[col].fillna('Unknown') if col in cats else X[col].fillna(X[col].median())

cw_arr  = compute_class_weight('balanced', classes=np.array([0,1]), y=y)
cw_dict = {0: float(cw_arr[0]), 1: float(cw_arr[1])}
spw     = cw_dict[1] / cw_dict[0]
print(f"✓ Class weight ratio: {spw:.2f}:1  (positive=surcharge)")
print(f"✓ Features: {len(avail)}")

unique_mh = df['Manhole_ID'].unique()
tv_mh, te_mh = train_test_split(unique_mh, test_size=TEST_FRAC, random_state=RANDOM_STATE)
tr_mh, va_mh  = train_test_split(tv_mh, test_size=VAL_FRAC/(1-TEST_FRAC), random_state=RANDOM_STATE)

def make_split(mh_list):
    m = df['Manhole_ID'].isin(mh_list)
    return X[m].reset_index(drop=True), y[m].reset_index(drop=True)

X_train, y_train = make_split(tr_mh)
X_val,   y_val   = make_split(va_mh)
X_test,  y_test  = make_split(te_mh)

assert not (set(tr_mh) & set(te_mh)), "Overlap!"
print(f"✓ Train:{len(X_train):,} | Val:{len(X_val):,} | Test:{len(X_test):,}")
print(f"  Train surge: {(y_train==1).mean()*100:.2f}%  "
      f"Test surge: {(y_test==1).mean()*100:.2f}%")

sw_train = compute_sample_weight('balanced', y_train)
sw_tv    = compute_sample_weight('balanced', pd.concat([y_train, y_val]).reset_index(drop=True))

checkpoint = load_checkpoint()
completed  = checkpoint['completed_models']
t_global   = datetime.now()
if completed: print(f"\n📌 Skipping: {', '.join(completed)}")

# ==============================================================================
# 1. LIGHTGBM
# ==============================================================================

if lgb is not None and 'LightGBM' not in completed:
    print("\n"+"="*70); print(f"[1/8] LIGHTGBM  ({N_FAST} trials)"); print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte = encode_cats_category(X_train,X_val,X_test,cats)

    def obj_lgb(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,1000),
               max_depth=trial.suggest_int('max_depth',3,15),
               learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
               num_leaves=trial.suggest_int('num_leaves',20,150),
               min_child_samples=trial.suggest_int('min_child_samples',5,100),
               subsample=trial.suggest_float('subsample',0.5,1.0),
               colsample_bytree=trial.suggest_float('colsample_bytree',0.5,1.0),
               reg_alpha=trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
               reg_lambda=trial.suggest_float('reg_lambda',1e-8,10.0,log=True),
               class_weight=cw_dict,random_state=RANDOM_STATE,n_jobs=-1,verbose=-1)
        sc,f1,rec,auc=objective_score(y_val,lgb.LGBMClassifier(**p).fit(Xtr,y_train).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc})
        if trial.number%10==0: print(f"  Trial {trial.number:>3}: F1={f1:.4f} Rec={rec:.4f} AUC={auc:.4f}")
        return sc

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_lgb,n_trials=N_FAST)
    bp={**st.best_params,'n_jobs':-1,'verbose':-1,'random_state':RANDOM_STATE,'class_weight':cw_dict}
    fm=lgb.LGBMClassifier(**bp).fit(pd.concat([Xtr,Xva]).reset_index(drop=True),
                                     pd.concat([y_train,y_val]).reset_index(drop=True))
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('LightGBM',y_test,(yp>=thr).astype(int),yp,m,thr)
    save_results('LightGBM',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_LightGBM'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'LightGBM',d)
    plot_threshold_analysis(y_test,yp,thr,'LightGBM',d)
    completed.append('LightGBM'); save_checkpoint(completed)

# ==============================================================================
# 2. XGBOOST
# ==============================================================================

if xgb is not None and 'XGBoost' not in completed:
    print("\n"+"="*70); print(f"[2/8] XGBOOST  ({N_FAST} trials)"); print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte=encode_cats_numeric(X_train,X_val,X_test,cats)

    def obj_xgb(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,1000),
               max_depth=trial.suggest_int('max_depth',3,15),
               learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
               subsample=trial.suggest_float('subsample',0.5,1.0),
               colsample_bytree=trial.suggest_float('colsample_bytree',0.5,1.0),
               gamma=trial.suggest_float('gamma',1e-8,1.0,log=True),
               reg_alpha=trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
               reg_lambda=trial.suggest_float('reg_lambda',1e-8,10.0,log=True),
               min_child_weight=trial.suggest_int('min_child_weight',1,10),
               scale_pos_weight=spw,random_state=RANDOM_STATE,n_jobs=-1,eval_metric='logloss')
        sc,f1,rec,auc=objective_score(y_val,xgb.XGBClassifier(**p).fit(Xtr,y_train).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc})
        if trial.number%10==0: print(f"  Trial {trial.number:>3}: F1={f1:.4f} Rec={rec:.4f} AUC={auc:.4f}")
        return sc

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_xgb,n_trials=N_FAST)
    bp={**st.best_params,'n_jobs':-1,'random_state':RANDOM_STATE,'scale_pos_weight':spw}
    fm=xgb.XGBClassifier(**bp).fit(pd.concat([Xtr,Xva]).reset_index(drop=True),
                                    pd.concat([y_train,y_val]).reset_index(drop=True))
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('XGBoost',y_test,(yp>=thr).astype(int),yp,m,thr)
    save_results('XGBoost',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_XGBoost'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'XGBoost',d)
    plot_threshold_analysis(y_test,yp,thr,'XGBoost',d)
    completed.append('XGBoost'); save_checkpoint(completed)

# ==============================================================================
# 3. CATBOOST
# ==============================================================================

if cb is not None and 'CatBoost' not in completed:
    print("\n"+"="*70); print(f"[3/8] CATBOOST  ({N_MEDIUM} trials)"); print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte=X_train.copy(),X_val.copy(),X_test.copy()
    for col in cats:
        for d in [Xtr,Xva,Xte]: d[col]=d[col].astype(str)

    def obj_cat(trial):
        p=dict(iterations=trial.suggest_int('iterations',100,1000),
               depth=trial.suggest_int('depth',3,10),
               learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
               l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1e-8,10.0,log=True),
               subsample=trial.suggest_float('subsample',0.5,1.0),
               cat_features=cats,auto_class_weights='Balanced',
               random_state=RANDOM_STATE,verbose=False,task_type='CPU',thread_count=-1)
        sc,f1,rec,auc=objective_score(y_val,cb.CatBoostClassifier(**p).fit(Xtr,y_train,verbose=False).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc})
        if trial.number%10==0: print(f"  Trial {trial.number:>3}: F1={f1:.4f} Rec={rec:.4f} AUC={auc:.4f}")
        return sc

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_cat,n_trials=N_MEDIUM)
    bp={**st.best_params,'cat_features':cats,'auto_class_weights':'Balanced',
        'thread_count':-1,'random_state':RANDOM_STATE}
    fm=cb.CatBoostClassifier(**bp).fit(pd.concat([Xtr,Xva]).reset_index(drop=True),
                                        pd.concat([y_train,y_val]).reset_index(drop=True),verbose=False)
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('CatBoost',y_test,(yp>=thr).astype(int),yp,m,thr)
    save_results('CatBoost',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_CatBoost'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'CatBoost',d)
    plot_threshold_analysis(y_test,yp,thr,'CatBoost',d)
    completed.append('CatBoost'); save_checkpoint(completed)

# ==============================================================================
# 4. RANDOM FOREST
# ==============================================================================

if 'Random Forest' not in completed:
    print("\n"+"="*70); print(f"[4/8] RANDOM FOREST  ({N_MEDIUM} trials)"); print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte=encode_cats_numeric(X_train,X_val,X_test,cats)

    def obj_rf(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,500),
               max_depth=trial.suggest_int('max_depth',10,50),
               min_samples_split=trial.suggest_int('min_samples_split',2,20),
               min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
               max_features=trial.suggest_categorical('max_features',['sqrt','log2',0.5,0.7]),
               class_weight='balanced',random_state=RANDOM_STATE,n_jobs=-1)
        sc,f1,rec,auc=objective_score(y_val,RandomForestClassifier(**p).fit(Xtr,y_train).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc})
        if trial.number%5==0: print(f"  Trial {trial.number:>3}: F1={f1:.4f} Rec={rec:.4f} AUC={auc:.4f}")
        return sc

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_rf,n_trials=N_MEDIUM)
    bp={**st.best_params,'class_weight':'balanced','n_jobs':-1,'random_state':RANDOM_STATE}
    fm=RandomForestClassifier(**bp).fit(pd.concat([Xtr,Xva]).reset_index(drop=True),
                                         pd.concat([y_train,y_val]).reset_index(drop=True))
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('Random Forest',y_test,(yp>=thr).astype(int),yp,m,thr)
    save_results('Random Forest',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_RandomForest'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'Random Forest',d)
    plot_threshold_analysis(y_test,yp,thr,'Random Forest',d)
    completed.append('Random Forest'); save_checkpoint(completed)

# ==============================================================================
# 5. GRADIENT BOOSTING
# ==============================================================================

if 'Gradient Boosting' not in completed:
    print("\n"+"="*70); print(f"[5/8] GRADIENT BOOSTING  ({N_SLOW} trials)"); print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte=encode_cats_numeric(X_train,X_val,X_test,cats)

    def obj_gb(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,500),
               max_depth=trial.suggest_int('max_depth',3,10),
               learning_rate=trial.suggest_float('learning_rate',0.01,0.3,log=True),
               min_samples_split=trial.suggest_int('min_samples_split',2,20),
               min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
               subsample=trial.suggest_float('subsample',0.5,1.0),
               max_features=trial.suggest_categorical('max_features',['sqrt','log2',None]),
               random_state=RANDOM_STATE)
        sc,f1,rec,auc=objective_score(y_val,GradientBoostingClassifier(**p).fit(Xtr,y_train,sample_weight=sw_train).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc})
        if trial.number%3==0:
            el=(datetime.now()-t0).total_seconds()
            print(f"  Trial {trial.number:>3}: F1={f1:.4f} Rec={rec:.4f} ETA {el/(trial.number+1)*(N_SLOW-trial.number-1)/60:.1f}min")
        return sc

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_gb,n_trials=N_SLOW)
    bp={**st.best_params,'random_state':RANDOM_STATE}
    fm=GradientBoostingClassifier(**bp).fit(pd.concat([Xtr,Xva]).reset_index(drop=True),
                                             pd.concat([y_train,y_val]).reset_index(drop=True),
                                             sample_weight=sw_tv)
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('Gradient Boosting',y_test,(yp>=thr).astype(int),yp,m,thr)
    save_results('Gradient Boosting',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_GradientBoosting'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'Gradient Boosting',d)
    plot_threshold_analysis(y_test,yp,thr,'Gradient Boosting',d)
    completed.append('Gradient Boosting'); save_checkpoint(completed)

# ==============================================================================
# 6. MLP
# ==============================================================================

if 'MLP' not in completed:
    print("\n"+"="*70); print(f"[6/8] MLP  ({N_SLOW} trials)"); print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte=encode_cats_numeric(X_train,X_val,X_test,cats)
    Xtr,Xva,Xte,sc_mlp=scale_numeric(Xtr,Xva,Xte,cats)

    def obj_mlp(trial):
        n_l=trial.suggest_int('n_layers',1,4)
        l_s=trial.suggest_categorical('layer_size',[64,128,256,512])
        p=dict(hidden_layer_sizes=tuple([l_s]*n_l),
               activation=trial.suggest_categorical('activation',['relu','tanh']),
               alpha=trial.suggest_float('alpha',1e-5,1.0,log=True),
               learning_rate_init=trial.suggest_float('learning_rate_init',1e-4,0.05,log=True),
               batch_size=trial.suggest_categorical('batch_size',[128,256,512]),
               max_iter=200,early_stopping=True,validation_fraction=0.1,
               n_iter_no_change=15,random_state=RANDOM_STATE)
        sc,f1,rec,auc=objective_score(y_val,MLPClassifier(**p).fit(Xtr,y_train,sample_weight=sw_train[:len(Xtr)]).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc})
        if trial.number%10==0: print(f"  Trial {trial.number:>3}: F1={f1:.4f} Rec={rec:.4f} AUC={auc:.4f}")
        return sc

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_mlp,n_trials=N_SLOW)
    best=st.best_params; n_l=best.pop('n_layers'); l_s=best.pop('layer_size')
    bp={**best,'hidden_layer_sizes':tuple([l_s]*n_l),'max_iter':500,
        'early_stopping':True,'validation_fraction':0.1,'n_iter_no_change':20,
        'random_state':RANDOM_STATE}
    Xtv=pd.concat([Xtr,Xva]).reset_index(drop=True)
    ytv=pd.concat([y_train,y_val]).reset_index(drop=True)
    fm=MLPClassifier(**bp).fit(Xtv,ytv,sample_weight=sw_tv)
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('MLP',y_test,(yp>=thr).astype(int),yp,m,thr)
    ts_str=datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUTPUT_DIR,f'MLP_{ts_str}.pkl'),'wb') as f:
        pickle.dump({'model':fm,'scaler':sc_mlp,'model_name':'MLP',
                     'best_params':bp,'metrics':m,'optimal_threshold':thr},f)
    save_results('MLP',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_MLP'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'MLP',d)
    plot_threshold_analysis(y_test,yp,thr,'MLP',d)
    completed.append('MLP'); save_checkpoint(completed)

# ==============================================================================
# 7. ADABOOST  — FIX: NO class_weight in DecisionTree (sample_weight handles imbalance)
# ==============================================================================

if 'AdaBoost' not in completed:
    print("\n"+"="*70); print(f"[7/8] ADABOOST  ({N_MEDIUM} trials)"); print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte=encode_cats_numeric(X_train,X_val,X_test,cats)

    def obj_ada(trial):
        bd=trial.suggest_int('base_max_depth',2,4)   # shallow = faster + less overfit
        p=dict(n_estimators=trial.suggest_int('n_estimators',50,500),
               learning_rate=trial.suggest_float('learning_rate',0.01,2.0,log=True),
               estimator=DecisionTreeClassifier(max_depth=bd,   # ← NO class_weight here
                                                random_state=RANDOM_STATE),
               random_state=RANDOM_STATE)
        sc,f1,rec,auc=objective_score(y_val,AdaBoostClassifier(**p).fit(Xtr,y_train,sample_weight=sw_train).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc,'base_depth':bd})
        if trial.number%5==0: print(f"  Trial {trial.number:>3}: F1={f1:.4f} Rec={rec:.4f} depth={bd}")
        return sc

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_ada,n_trials=N_MEDIUM)
    best=st.best_params; bd=best.pop('base_max_depth')
    bp={**best,
        'estimator': DecisionTreeClassifier(max_depth=bd,   # ← NO class_weight here
                                             random_state=RANDOM_STATE),
        'random_state': RANDOM_STATE}
    fm=AdaBoostClassifier(**bp).fit(pd.concat([Xtr,Xva]).reset_index(drop=True),
                                     pd.concat([y_train,y_val]).reset_index(drop=True),
                                     sample_weight=sw_tv)
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('AdaBoost',y_test,(yp>=thr).astype(int),yp,m,thr)
    save_results('AdaBoost',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_AdaBoost'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'AdaBoost',d)
    plot_threshold_analysis(y_test,yp,thr,'AdaBoost',d)
    completed.append('AdaBoost'); save_checkpoint(completed)

# ==============================================================================
# 8. SVC
# ==============================================================================

if 'SVC' not in completed:
    print("\n"+"="*70)
    print(f"[8/8] SVC  ({N_SVC} trials — subsampled: trial={SVC_TRIAL_N:,} final={SVC_FINAL_N:,})")
    print("="*70)
    t0=datetime.now(); hist=[]
    Xtr,Xva,Xte=encode_cats_numeric(X_train,X_val,X_test,cats)
    Xtr,Xva,Xte,sc_svc=scale_numeric(Xtr,Xva,Xte,cats)
    Xtr_s,ytr_s,sw_s=subsample_rows(Xtr,y_train,SVC_TRIAL_N,sw=sw_train)

    def obj_svc(trial):
        p=dict(kernel=trial.suggest_categorical('kernel',['rbf','poly','sigmoid']),
               C=trial.suggest_float('C',0.1,100.0,log=True),
               gamma=trial.suggest_categorical('gamma',['scale','auto']),
               class_weight='balanced',probability=True,random_state=RANDOM_STATE)
        if p['kernel']=='poly': p['degree']=trial.suggest_int('degree',2,4)
        sc_v,f1,rec,auc=objective_score(y_val,SVC(**p).fit(Xtr_s,ytr_s,sample_weight=sw_s).predict_proba(Xva)[:,1])
        hist.append({'trial':trial.number,'f1':f1,'recall':rec,'roc_auc':auc,'combined':sc_v})
        if trial.number%5==0: print(f"  Trial {trial.number:>3}: F1={f1:.4f} kernel={p['kernel']}")
        return sc_v

    st=optuna.create_study(direction='maximize',sampler=TPESampler(seed=RANDOM_STATE),pruner=MedianPruner())
    st.optimize(obj_svc,n_trials=N_SVC)
    bp={**st.best_params,'class_weight':'balanced','probability':True,'random_state':RANDOM_STATE}
    Xtv=pd.concat([Xtr,Xva]).reset_index(drop=True)
    ytv=pd.concat([y_train,y_val]).reset_index(drop=True)
    sw_tvf=compute_sample_weight('balanced',ytv)
    Xtv_s,ytv_s,sw_tvs=subsample_rows(Xtv,ytv,SVC_FINAL_N,sw=sw_tvf)
    print(f"  Training final SVC on {len(Xtv_s):,} rows …")
    fm=SVC(**bp).fit(Xtv_s,ytv_s,sample_weight=sw_tvs)
    yp=fm.predict_proba(Xte)[:,1]; thr,_=find_optimal_threshold(y_test,yp)
    m=compute_metrics(y_test,(yp>=thr).astype(int),yp,(datetime.now()-t0).total_seconds())
    print_metrics('SVC',y_test,(yp>=thr).astype(int),yp,m,thr)
    ts_str=datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(OUTPUT_DIR,f'SVC_{ts_str}.pkl'),'wb') as f:
        pickle.dump({'model':fm,'scaler':sc_svc,'model_name':'SVC',
                     'best_params':bp,'metrics':m,'optimal_threshold':thr},f)
    save_results('SVC',fm,bp,m,thr,hist)
    d=os.path.join(OUTPUT_DIR,'Diagnostics_SVC'); os.makedirs(d,exist_ok=True)
    plot_confusion_matrix(y_test,(yp>=thr).astype(int),'SVC',d)
    plot_threshold_analysis(y_test,yp,thr,'SVC',d)
    completed.append('SVC'); save_checkpoint(completed)

# ==============================================================================
# JOINT DIAGNOSTICS
# ==============================================================================

print("\n"+"="*70); print("JOINT DIAGNOSTICS"); print("="*70)

import glob

sf  = os.path.join(OUTPUT_DIR, 'classification_summary.csv')
sdf = (pd.read_csv(sf)
         .sort_values('Combined', ascending=False)
         .drop_duplicates(subset='Model', keep='first')   # ← dedup: keep best run per model
         .reset_index(drop=True))

models_data_joint = []
for _, row in sdf.iterrows():
    name    = row['Model']
    pattern = os.path.join(OUTPUT_DIR, f"{name.replace(' ','_')}_*.pkl")
    pkls    = [p for p in glob.glob(pattern) if not any(x in p for x in ['params','trials'])]
    if not pkls: continue
    with open(sorted(pkls, key=os.path.getmtime)[-1], 'rb') as f: pkg = pickle.load(f)
    model = pkg['model']; thr = pkg['optimal_threshold']

    name_l = name.lower()
    if name_l == 'lightgbm':
        Xtr_,Xva_,Xte_ = encode_cats_category(X_train,X_val,X_test,cats)
    elif name_l == 'catboost':
        Xtr_,Xva_,Xte_ = X_train.copy(),X_val.copy(),X_test.copy()
        for col in cats:
            for d in [Xtr_,Xva_,Xte_]: d[col]=d[col].astype(str)
    elif name_l in ('mlp','svc'):
        Xtr_,Xva_,Xte_ = encode_cats_numeric(X_train,X_val,X_test,cats)
        sc_ = pkg.get('scaler')
        if sc_ is not None:
            num_cols = [c for c in Xtr_.columns if c not in cats]
            Xtr_[num_cols]=sc_.transform(Xtr_[num_cols])
            Xva_[num_cols]=sc_.transform(Xva_[num_cols])
            Xte_[num_cols]=sc_.transform(Xte_[num_cols])
    else:
        Xtr_,Xva_,Xte_ = encode_cats_numeric(X_train,X_val,X_test,cats)

    yp_ = model.predict_proba(Xte_)[:,1]
    models_data_joint.append((name, y_test, yp_))

if models_data_joint:
    plot_roc_pr_curves(models_data_joint, OUTPUT_DIR)
    plot_calibration(models_data_joint,   OUTPUT_DIR)

plot_model_comparison(sdf, OUTPUT_DIR)

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print(f"\n{'='*70}")
print("FINAL RESULTS — 8-MODEL CLASSIFICATION")
print(f"{'='*70}")
print(f"\n  {'#':<3} {'Model':<22} {'Combined':>9} {'AUC':>8} {'AvgPrec':>9} "
      f"{'F1':>7} {'Recall':>8} {'Precis':>8} {'Brier':>7}")
print("  "+"─"*84)
medals = {0:'🏆', 1:'🥈', 2:'🥉'}
for i, (_, row) in enumerate(sdf.iterrows()):
    sym = medals.get(i, '   ')
    print(f"  {sym} {row['Model']:<22} {row['Combined']:>9.4f} {row['ROC_AUC']:>8.4f} "
          f"{row['Avg_Precision']:>9.4f} {row['F1']:>7.4f} {row['Recall']:>8.4f} "
          f"{row['Precision']:>8.4f} {row['Brier']:>7.4f}")

# ==============================================================================
# FULL SHAP — TOP-2 MODELS
# ==============================================================================

if shap is None:
    print("\n⚠ SHAP not installed — skipping"); raise SystemExit()

top2_names = ['XGBoost', 'LightGBM', 'Gradient Boosting']
print(f"\n{'='*70}")
print(f"FULL SHAP — TOP-2: {top2_names[0]}  &  {top2_names[1]}")
print(f"{'='*70}")

TREE_SET = {'lightgbm','xgboost','catboost','random forest','gradient boosting','adaboost'}

for model_name in top2_names:
    pattern = os.path.join(OUTPUT_DIR, f"{model_name.replace(' ','_')}_*.pkl")
    pkls    = [p for p in glob.glob(pattern) if not any(x in p for x in ['params','trials'])]
    if not pkls:
        print(f"⚠ No pkl for {model_name} — skipping"); continue
    with open(sorted(pkls, key=os.path.getmtime)[-1], 'rb') as f: pkg = pickle.load(f)
    model = pkg['model']; thr = pkg['optimal_threshold']; m = pkg['metrics']

    name_l = model_name.lower()
    if name_l == 'lightgbm':
        Xtr_,Xva_,Xte_ = encode_cats_category(X_train,X_val,X_test,cats)
    elif name_l == 'catboost':
        Xtr_,Xva_,Xte_ = X_train.copy(),X_val.copy(),X_test.copy()
        for col in cats:
            for d in [Xtr_,Xva_,Xte_]: d[col]=d[col].astype(str)
    elif name_l in ('mlp','svc'):
        Xtr_,Xva_,Xte_ = encode_cats_numeric(X_train,X_val,X_test,cats)
        sc_ = pkg.get('scaler')
        if sc_ is not None:
            num_cols = [c for c in Xtr_.columns if c not in cats]
            Xtr_[num_cols]=sc_.transform(Xtr_[num_cols])
            Xva_[num_cols]=sc_.transform(Xva_[num_cols])
            Xte_[num_cols]=sc_.transform(Xte_[num_cols])
    else:
        Xtr_,Xva_,Xte_ = encode_cats_numeric(X_train,X_val,X_test,cats)

    yp_ = model.predict_proba(Xte_)[:,1]
    run_full_shap_analysis(model=model, model_name=model_name,
                           X_te=Xte_, y_te=y_test, y_proba=yp_,
                           threshold=thr, metrics=m, cat_cols=cats,
                           X_tr=Xtr_, is_tree=(name_l in TREE_SET))

if os.path.exists(CHECKPOINT_FILE): os.remove(CHECKPOINT_FILE)

print(f"\n{'='*70}"); print("✅ ALL DONE"); print(f"{'='*70}")
print(f"\n📁 {OUTPUT_DIR}")
print(f"\n  Per model  : Diagnostics_<Model>/")
print(f"  Joint plots: roc_pr_curves.png | calibration_curve.png | model_comparison_8models.png")
print(f"  SHAP top-2 : SHAP_{top2_names[0].replace(' ','_')}/ | SHAP_{top2_names[1].replace(' ','_')}/")