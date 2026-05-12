"""
01_maintenance_classification.py
==================================
Classifies maintenance work order activities into 14 categories,
computes severity and response time metrics, and aggregates results
to per-feature (pipe) level for downstream manhole-level joining.

Categories:
  Blockages, Blockage and Inspection, General Cleaning,
  Infiltration and inflow, Flow velocity/hydraulic conditions,
  Cracks and fractures, Defective lining, Deformation,
  Corrosion and abrasion of pipes, Defective connection,
  Broken manhole covers/damaged manhole walls,
  Environmental, Inspection, General Maintenance

Input:  CLeanedAllSheets.xlsx  (from script 00)
Output: combined_statistics.xlsx + per-sheet CSVs in by_sheet/

Author: Research Team
"""

import pandas as pd
import numpy as np
import re
import os
import sys
from datetime import datetime
from collections import defaultdict
import warnings
import json
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import psutil

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

# ============================================
# CONFIGURATION
# ============================================

class Config:
    INPUT_FILE  = str(Paths.CLEANED_MAINTENANCE)
    OUTPUT_DIR  = str(Paths.MAINTENANCE_RESULTS)

    SHEETS_TO_PROCESS = [
        "Export Worksheet",
        "Sheet1", "Sheet2", "Sheet3", "Sheet4", "Sheet5"
    ]

    COLUMN_MAPPING = {
        'feature_number':       'Feature Number',
        'description':          'Description of Works',
        'date_of_issue':        'Date of Issue',
        'date_of_commencement': 'Date of Commencement',
        'date_of_completion':   'Date of Completion',
        'contract_no':          'Contract No.',
        'type_of_works':        'Type of Works',
        'location':             'Location',
        'estimated_value':      'Estimated Value of Works',
        'certified_amount':     'Certified Amount',
        'to_number':            'T.O. Number or EIMS File Ref. No.'
    }

    CHUNK_SIZE            = Params.CHUNK_SIZE
    MAX_WORKERS           = max(1, int(cpu_count() * 0.75))
    SPLIT_ACTIVITIES      = True
    SPLIT_ON_NEWLINE_ONLY = True
    MIN_ACTIVITY_LENGTH   = Params.MIN_ACTIVITY_LENGTH

    GENERATE_BY_SHEET     = True
    GENERATE_BY_PREFIX    = False
    ENABLE_CHECKPOINTING  = True
    VERBOSE               = True
    MEMORY_WARNING_GB     = 30

    EXCEL_ROW_LIMIT       = Params.EXCEL_ROW_LIMIT
    EXCEL_SHEET_NAME_LIMIT = 31

    RESPONSE_TIME_CATEGORIES = Params.RESPONSE_TIME_CATEGORIES


# ============================================
# PROGRESS & MONITORING
# ============================================

class ProgressTracker:
    def __init__(self):
        self.start = time.time()
        self.sheets = {}

    def start_sheet(self, name, rows):
        self.sheets[name] = {'start': time.time(), 'total': rows, 'done': 0}

    def update(self, name, rows):
        if name in self.sheets:
            s = self.sheets[name]
            s['done'] += rows
            pct  = s['done'] / s['total'] * 100
            rate = s['done'] / (time.time() - s['start'])
            eta  = (s['total'] - s['done']) / rate / 60 if rate > 0 else 0
            print(f"  [{name}] {pct:.1f}% | {s['done']:,}/{s['total']:,} | "
                  f"{rate:,.0f} rows/s | ETA: {eta:.1f}m")

    def done(self, name):
        if name in self.sheets:
            t = (time.time() - self.sheets[name]['start']) / 60
            print(f"  ✅ {name} done in {t:.1f}m")


class MemoryMonitor:
    @staticmethod
    def check(threshold_gb):
        gb = psutil.Process().memory_info().rss / (1024 ** 3)
        if gb > threshold_gb:
            print(f"  ⚠️  Memory: {gb:.2f}GB")
        return gb


# ============================================
# CATEGORIZATION ENGINE
# ============================================

class CategoryEngine:
    def __init__(self):
        self.hierarchy = [
            "Blockages",
            "Blockage and Inspection",
            "General Cleaning",
            "Infiltration and inflow",
            "Flow velocity/ hydraulic conditions",
            "Cracks and fractures",
            "Defective lining",
            "Deformation",
            "Corrosion and abrasion of pipes",
            "Defective connection",
            "Broken manhole covers/ damaged manhole walls",
            "Environmental",
            "Inspection",
            "General Maintenance"
        ]

        self.categories = {
            "Blockages": {
                'primary': [
                    "chok", "block", "clog", "obstruct", "clear",
                    ("clear", "block"), ("remove", "block"), ("clear", "chok"),
                    ("remove", "chok"), ("clear", "obstruction"), ("remove", "obstruction"),
                    ("clear", "debris"), ("remove", "debris"), ("clear", "silt"),
                    ("remove", "silt"), ("clear", "sediment"), ("remove", "sediment"),
                    ("clear", "grease"), ("remove", "grease"), ("clear", "root"),
                    ("remove", "root"), ("clear", "hard"), ("remove", "hard"),
                    "desilt", "jet", "flush", "hydro"
                ],
                'secondary': [
                    "flush", "silt", "sediment", "grease", "root", "debris",
                    "obstruction", "screen", "desilt"
                ]
            },
            "Blockage and Inspection": {
                'primary': [
                    ("clear", "cctv"), ("flush", "cctv"),
                    ("jet", "cctv"), ("desilt", "cctv"), ("inspect", "block"),
                    ("inspect", "chok"), ("survey", "block"), ("survey", "chok"),
                    ("cctv", "block"), ("cctv", "chok"), ("inspect", "clear"),
                    ("survey", "clear"), ("monitor", "block"), ("diagnose", "block")
                ],
                'secondary': []
            },
            "General Cleaning": {
                'primary': [
                    "clean", "cleans", "cleansing", "cleaning", "cleanse", "wash",
                    ("clean", "sewer"), ("clean", "pipeline"), ("clean", "drain"),
                    ("clean", "SWD"), ("cleanse", "sewer"), ("cleanse", "pipeline"),
                    ("cleanse", "drain"), ("cleanse", "SWD"), ("wash", "sewer"),
                    ("wash", "drain"), ("wash", "pipeline"), ("routine", "clean"),
                    ("preventive", "clean"), ("maintenance", "clean")
                ],
                'secondary': [
                    "clear", "clean", "cleans", "cleansing", "cleaning", "cleanse",
                    "wash", "routine", "preventive", "maintenance"
                ]
            },
            "Cracks and fractures": {
                'primary': [
                    "crack", "fracture", "split", "break", "fissure",
                    ("seal", "crack"), ("repair", "crack"), ("leak", "crack"),
                    ("fill", "void"), ("rapid", "concrete")
                ],
                'secondary': [
                    "defect", "damage", "rupture", "gap", "seep", "mortar",
                    ("culvert", "repair"), ("repair", "sewer"),
                    ("repair", "drain"), ("repair", "channel"),
                    ("repair", "pipe"), ("repair", "structural")
                ]
            },
            "Infiltration and inflow": {
                'primary': [
                    "infiltration", "inflow", "I&I",
                    ("seal", "joint"), ("seal", "leak"), ("water", "ingress"),
                    ("stop", "leak"), ("grout", "joint"), ("seal", "hole")
                ],
                'secondary': [
                    "leak", "seep", "seepage", "wet", "drip", "groundwater"
                ]
            },
            "Defective lining": {
                'primary': [
                    "lining", "liner", "CIPP", "reline", "sleeve",
                    ("defective", "lining"), ("failed", "lining"), ("robot", "cut"),
                    ("robotic", "cut"), ("robotic", "machine"), ("cutting", "machine"),
                    ("structural", "improvement"), ("structural", "rehabilitation"),
                    ("structural", "repair"), ("structural", "upgrade"),
                    ("pipe", "rehabilitation"), ("sewer", "rehabilitation"),
                    ("drain", "rehabilitation"), ("pipeline", "rehabilitation")
                ],
                'secondary': [
                    "epoxy", "resin", "cure", "rehab", "robot", "robotic", "cutting"
                ]
            },
            "Corrosion and abrasion of pipes": {
                'primary': [
                    "corros", "abrad", "eros", "pitt", "rust",
                    ("chemical", "attack"), ("protective", "coat"), ("surface", "coat")
                ],
                'secondary': [
                    "deteriorate", "wear", "thin", "metal loss", "scale", "deposit",
                    "mortar", "mort", ("spray", "concrete")
                ]
            },
            "Deformation": {
                'primary': [
                    "deform", "buckl", "bent", "oval", "sag", "deflect", "jack",
                    "surround", ("pipe", "collapse"), ("distort", "pipe"),
                    ("excavat", "repair"), ("excavat", "damage"), ("excavat", "sewer"),
                    ("collapse", "pipe"), ("collapse", "sewer"), ("spray", "concrete"),
                    "excavat", "excavate",
                    ("trench", "damage"), ("trench", "pipe"), ("trench", "SWD"),
                    ("excavat", "trench"), ("trial", "trench"),
                    ("locate", "damage"), ("locate", "pipe")
                ],
                'secondary': [
                    "misalign", "settle", "subsidence", "shift", "displace",
                    ("replace", "SWD"), ("replace", "FWD"), ("fill", "concrete"),
                    ("lay", "pipe"), ("structural", "repair"), ("trench", "pipe"),
                    ("expose", "repair"), ("excavate", "soft"), "trench", "trial"
                ]
            },
            "Defective connection": {
                'primary': [
                    "joint", "junction", "tee", "lateral", "tapping",
                    "abandon", "disused", "illegal",
                    ("defective", "joint"), ("leaking", "connection")
                ],
                'secondary': [
                    "fitting", "tap", "branch", "disconnected", "offset"
                ]
            },
            "Broken manhole covers/ damaged manhole walls": {
                'primary': [
                    "cover", "frame", "grate", "lid", "recess", "manhole", "chamber",
                    ("broken", "manhole"), ("damaged", "cover"), ("repair", "manhole"),
                    ("replace", "cover"), ("broken", "cover"), ("damaged", "manhole"),
                    ("crack", "manhole"), ("fracture", "manhole"), ("seal", "manhole"),
                    "m/h", "man hole"
                ],
                'secondary': []
            },
            "Environmental": {
                'primary': [
                    "vegetation", "tree", "root", "planting", "grass",
                    ("cut", "vegetation"), ("remove", "tree"), ("cut", "grass")
                ],
                'secondary': [
                    "soil", "erosion", "slope", "nullah", "channel", "river"
                ]
            },
            "Flow velocity/ hydraulic conditions": {
                'primary': [
                    "flow", "hydraulic", "velocity", "capacity", "surcharge", "valve",
                    ("improve", "flow"), ("modify", "intake"), ("debris", "intake"),
                    ("remove", "river"), ("repair", "river")
                ],
                'secondary': [
                    "gradient", "slope", "discharge", "overflow", "backwater"
                ]
            },
            "Inspection": {
                'primary': [
                    ("carry", "CCTV"), ("color", "survey"), ("conduct", "cctv"),
                    ("conduct", "survey"), ("trial", "trench"), ("locate", "damage"),
                    ("locate", "pipe"), ("excavat", "trial"), ("excavat", "inspect"),
                    ("provide", "report"), ("provide", "photo")
                ],
                'secondary': [
                    "record", "report", "assess", "review", "check", "trial trench",
                    "CCTV", "survey", "inspect", "monitor", "diagnose", "location",
                    "locate", "identify", "find"
                ]
            },
            "General Maintenance": {
                'primary': [],
                'secondary': []
            }
        }

        self.severity_kw = {
            'EMERGENCY': ["emergency", "urgent", "critical", "collapse", "burst",
                          "flooding", "hazard"],
            'HIGH':   ["major", "severe", "serious", "failure", "rupture"],
            'MEDIUM': ["repair", "fix", "maintenance", "cleaning", "minor"],
            'LOW':    ["routine", "preventive", "inspection", "monitoring"]
        }

        self.compile()

    def compile(self):
        def stem(word):
            s = r'(?:ed|ing|en|s|age|y|ies|ion|al|ive|ic|sing|ment|er|or|ly|able|ful|less|ness|ish|ize|ise)?'
            return rf'\b{re.escape(word)}{s}\b'

        for cat, data in self.categories.items():
            data['p_re'] = []
            for kw in data['primary']:
                if isinstance(kw, tuple):
                    data['p_re'].append(
                        re.compile(rf'(?=.*{stem(kw[0])})(?=.*{stem(kw[1])})', re.I))
                else:
                    data['p_re'].append(re.compile(stem(kw), re.I))

            data['s_re'] = []
            for kw in data['secondary']:
                if isinstance(kw, tuple):
                    data['s_re'].append(
                        re.compile(rf'(?=.*{stem(kw[0])})(?=.*{stem(kw[1])})', re.I))
                else:
                    data['s_re'].append(re.compile(stem(kw), re.I))

        for lvl, kws in self.severity_kw.items():
            self.severity_kw[lvl] = [re.compile(rf'\b{re.escape(k)}\b', re.I) for k in kws]

    def categorize(self, text):
        if isinstance(text, str) and text.strip().lower() == "ditto":
            return None

        txt  = text.lower()
        prim, sec = {}, {}

        # Manhole-first logic
        manhole_patterns = [
            r'man.?hole', r'm[/\s]?h', r'chamber', r'cover',
            r'frame', r'grate', r'lid', r'recess'
        ]
        has_manhole = any(re.search(p, txt, re.I) for p in manhole_patterns)
        if has_manhole:
            blockage_terms = ['block', 'chok', 'clog', 'obstruct',
                              'clear', 'flush', 'jet', 'hydro', 'desilt']
            if not any(t in txt for t in blockage_terms):
                return {'cat': 'Broken manhole covers/ damaged manhole walls',
                        'all': ['Broken manhole covers/ damaged manhole walls'],
                        'conf': 100, 'prim': True}

        # Blockage and Inspection (highest priority compound)
        for r in self.categories["Blockage and Inspection"]['p_re']:
            if r.search(txt):
                prim["Blockage and Inspection"] = True
                break

        if "Blockage and Inspection" not in prim:
            for cat, data in self.categories.items():
                if cat == "Blockage and Inspection":
                    continue
                for r in data['p_re']:
                    if r.search(txt):
                        prim[cat] = True
                        break

        for cat, data in self.categories.items():
            if cat not in prim:
                for r in data['s_re']:
                    if r.search(txt):
                        sec[cat] = True
                        break

        # Deconflict General Cleaning vs Blockages
        if "General Cleaning" in prim and "Blockages" in prim:
            del prim["General Cleaning"]

        for collection in (prim, sec):
            if "Blockage and Inspection" in collection:
                collection.pop("Blockages",   None)
                collection.pop("Inspection",  None)

        if prim:
            cats = sorted(prim.keys(),
                          key=lambda c: self.hierarchy.index(c)
                          if c in self.hierarchy else 999)
            return {'cat': cats[0], 'all': cats, 'conf': 100, 'prim': True}

        if sec:
            cats = sorted(sec.keys(),
                          key=lambda c: self.hierarchy.index(c)
                          if c in self.hierarchy else 999)
            return {'cat': cats[0], 'all': cats, 'conf': 60, 'prim': False}

        return {'cat': 'General Maintenance', 'all': ['General Maintenance'],
                'conf': 0, 'prim': False}

    def severity(self, text):
        if isinstance(text, str) and text.strip().lower() == "ditto":
            return "DITTO"
        txt = text.lower()
        for lvl in ['EMERGENCY', 'HIGH', 'MEDIUM', 'LOW']:
            for r in self.severity_kw[lvl]:
                if r.search(txt):
                    return f"{lvl} - Keyword"
        return "ROUTINE - Keyword"


# ============================================
# TIME ANALYZER
# ============================================

class TimeAnalyzer:
    def __init__(self, config):
        self.config = config

    def parse_dates(self, series):
        strs = series.astype(str).str.split(' ').str[0]
        try:
            return pd.to_datetime(strs, format='%d/%m/%Y', errors='coerce')
        except Exception:
            return pd.to_datetime(strs, errors='coerce', dayfirst=True)

    def response_time(self, issue, commence):
        i    = self.parse_dates(issue)
        c    = self.parse_dates(commence)
        days = (c - i).dt.days
        days[days < 0] = None
        return days

    def severity_from_days(self, days):
        if pd.isna(days) or days < 0:
            return 'Unknown'
        for sev, thresh in self.config.RESPONSE_TIME_CATEGORIES.items():
            if days <= thresh:
                return sev
        return 'Backlog - Over 1 Month'

    def extract_year(self, series):
        return self.parse_dates(series).dt.year


# ============================================
# ACTIVITY PARSER
# ============================================

class ActivityParser:
    def __init__(self, config):
        self.config = config

    def parse(self, desc):
        if not isinstance(desc, str) or len(desc) < self.config.MIN_ACTIVITY_LENGTH:
            return [str(desc)]
        if not self.config.SPLIT_ACTIVITIES:
            return [desc]
        normalized = re.sub(r'\r\n|\r', '\n', desc.strip())
        parts = [p.strip() for p in normalized.split('\n')
                 if p.strip() and len(p.strip()) >= self.config.MIN_ACTIVITY_LENGTH]
        return parts if len(parts) > 1 else [desc]


# ============================================
# WORKER FUNCTION
# ============================================

def process_chunk(args):
    chunk, config = args
    parser  = ActivityParser(config)
    cat_eng = CategoryEngine()
    results = []

    for idx, row in chunk.iterrows():
        acts = parser.parse(row.get('description', ''))
        last_category = None

        for act_idx, act in enumerate(acts, 1):
            if len(str(act).strip()) < config.MIN_ACTIVITY_LENGTH:
                continue

            if isinstance(act, str) and act.strip().lower() == "ditto":
                if last_category is not None:
                    cat_result = last_category.copy()
                    sev = "DITTO"
                else:
                    cat_result = {'cat': 'General Maintenance',
                                  'all': ['General Maintenance'], 'conf': 0, 'prim': False}
                    sev = cat_eng.severity(act)
            else:
                cat_result = cat_eng.categorize(act)
                sev = cat_eng.severity(act)
                if cat_result is not None:
                    last_category = cat_result.copy()

            if cat_result is None:
                cat_result = (last_category.copy() if last_category
                              else {'cat': 'General Maintenance',
                                    'all': ['General Maintenance'],
                                    'conf': 0, 'prim': False})

            results.append({
                'feature_number':            row.get('feature_number', ''),
                'feature_prefix':            row.get('feature_prefix', ''),
                'activity_id':               f"{row.get('feature_number', '')}_{act_idx}",
                'activity_index':            act_idx,
                'total_activities_in_record': len(acts),
                'activity_text':             act,
                'primary_category':          cat_result['cat'],
                'all_categories':            ', '.join(cat_result['all']),
                'categorization_confidence': cat_result['conf'],
                'primary_keyword_match':     cat_result['prim'],
                'severity_keyword_based':    sev,
                'response_time_days':        row.get('response_time_days'),
                'severity_response_based':   row.get('severity_response_based', ''),
                'date_of_issue':             row.get('date_of_issue'),
                'date_of_commencement':      row.get('date_of_commencement'),
                'date_of_completion':        row.get('date_of_completion'),
                'completion_year':           row.get('completion_year'),
                'contract_no':               row.get('contract_no', ''),
                'type_of_works':             row.get('type_of_works', ''),
                'location':                  row.get('location', ''),
                'estimated_value':           row.get('estimated_value'),
                'certified_amount':          row.get('certified_amount'),
                'to_number':                 row.get('to_number', ''),
                'source_sheet':              row.get('source_sheet', ''),
                'original_row_index':        idx
            })

    return results


# ============================================
# MAIN PROCESSOR
# ============================================

class SewerProcessor:
    def __init__(self, config=None):
        self.config        = config or Config()
        self.time_analyzer = TimeAnalyzer(self.config)
        self.progress      = ProgressTracker()
        self.feature_history  = []
        self.activity_details = []

    def get_prefix(self, fn):
        if not isinstance(fn, str) or not fn.strip():
            return 'OTHER'
        for c in fn.strip():
            if c.isalpha():
                return c.upper()
        return 'OTHER'

    def checkpoint_exists(self, sheet):
        cp = os.path.join(self.config.OUTPUT_DIR, 'checkpoints',
                          f'{sheet}_checkpoint.json')
        return os.path.exists(cp)

    def save_checkpoint(self, sheet, stats):
        os.makedirs(os.path.join(self.config.OUTPUT_DIR, 'checkpoints'),
                    exist_ok=True)
        cp = os.path.join(self.config.OUTPUT_DIR, 'checkpoints',
                          f'{sheet}_checkpoint.json')
        with open(cp, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

    def build_feature_history(self, df, sheet_name):
        records = []
        for _, row in df.iterrows():
            records.append({
                'feature_number':  row.get('feature_number', ''),
                'feature_prefix':  row.get('feature_prefix', ''),
                'date_of_completion': row.get('date_of_completion'),
                'source_sheet':    sheet_name,
                'estimated_value': row.get('estimated_value', 0)
            })
        return records

    def process_sheet(self, sheet_name, df):
        print(f"\n{'='*70}\nProcessing: {sheet_name}\n{'='*70}")

        if self.config.ENABLE_CHECKPOINTING and self.checkpoint_exists(sheet_name):
            print(f"  ✅ {sheet_name} done (checkpoint). Rebuilding history only...")
            df['source_sheet']    = sheet_name
            df['feature_prefix'] = df[self.config.COLUMN_MAPPING['feature_number']].apply(
                self.get_prefix)
            for k, v in self.config.COLUMN_MAPPING.items():
                if v in df.columns and k != v:
                    df[k] = df[v]
            self.feature_history.extend(self.build_feature_history(df, sheet_name))
            csv_path = os.path.join(self.config.OUTPUT_DIR, 'by_sheet',
                                    f'{sheet_name}_activities.csv')
            if os.path.exists(csv_path):
                self.activity_details.append(pd.read_csv(csv_path))
            return

        self.progress.start_sheet(sheet_name, len(df))
        df['source_sheet']    = sheet_name
        df['feature_prefix'] = df[self.config.COLUMN_MAPPING['feature_number']].apply(
            self.get_prefix)

        print("  🔄 Parsing dates...")
        df['response_time_days'] = self.time_analyzer.response_time(
            df[self.config.COLUMN_MAPPING['date_of_issue']],
            df[self.config.COLUMN_MAPPING['date_of_commencement']]
        )
        df['severity_response_based'] = df['response_time_days'].apply(
            self.time_analyzer.severity_from_days)
        df['completion_year'] = self.time_analyzer.extract_year(
            df[self.config.COLUMN_MAPPING['date_of_completion']])

        for k, v in self.config.COLUMN_MAPPING.items():
            if v in df.columns and k != v:
                df[k] = df[v]

        self.feature_history.extend(self.build_feature_history(df, sheet_name))

        all_results = []
        chunks = np.array_split(df, max(1, len(df) // self.config.CHUNK_SIZE + 1))
        print(f"  🔄 Processing {len(chunks)} chunks with "
              f"{self.config.MAX_WORKERS} workers...")

        for chunk in chunks:
            workers   = np.array_split(chunk, self.config.MAX_WORKERS)
            args      = [(w, self.config) for w in workers if len(w) > 0]
            with Pool(self.config.MAX_WORKERS) as pool:
                chunk_res = pool.map(process_chunk, args)
            for r in chunk_res:
                all_results.extend(r)
            self.progress.update(sheet_name, len(chunk))
            MemoryMonitor.check(self.config.MEMORY_WARNING_GB)

        result_df = pd.DataFrame(all_results)
        self.activity_details.append(result_df)
        self.save_outputs(sheet_name, result_df)

        stats = {'status': 'complete', 'rows': len(df),
                 'activities': len(result_df), 'time': datetime.now().isoformat()}
        if self.config.ENABLE_CHECKPOINTING:
            self.save_checkpoint(sheet_name, stats)
        self.progress.done(sheet_name)

    def save_outputs(self, sheet, df):
        by_sheet = os.path.join(self.config.OUTPUT_DIR, 'by_sheet')
        os.makedirs(by_sheet, exist_ok=True)
        if self.config.GENERATE_BY_SHEET:
            df.to_csv(os.path.join(by_sheet, f'{sheet}_activities.csv'), index=False)
            print(f"  ✅ Saved sheet CSV")

    def build_enhanced_feature_history(self, all_activities_df):
        key_categories = [
            "Blockages", "Blockage and Inspection", "General Cleaning",
            "Infiltration and inflow", "Flow velocity/ hydraulic conditions",
            "Cracks and fractures", "Defective lining", "Deformation",
            "Corrosion and abrasion of pipes", "Defective connection",
            "Broken manhole covers/ damaged manhole walls",
            "Environmental", "Inspection", "General Maintenance"
        ]
        severity_levels    = ['EMERGENCY', 'HIGH', 'MEDIUM', 'LOW', 'ROUTINE', 'DITTO']
        response_cats      = list(self.config.RESPONSE_TIME_CATEGORIES.keys())
        enhanced_history   = []

        for feature_num, group in all_activities_df.groupby('feature_number'):
            feat = {
                'feature_number': feature_num,
                'feature_prefix': group['feature_prefix'].iloc[0]
                if 'feature_prefix' in group.columns else 'OTHER',
                'total_activities': len(group),
            }
            for cat in key_categories:
                cnt       = (group['primary_category'] == cat).sum()
                clean_cat = cat.replace('/', '_').replace(' ', '_').replace('&', 'and')
                feat[f'{clean_cat}_count'] = cnt
                feat[f'{clean_cat}_pct']   = round(cnt / len(group) * 100, 2) if group.shape[0] else 0

            for sev in severity_levels:
                if 'severity_keyword_based' in group.columns:
                    cnt = group['severity_keyword_based'].str.contains(
                        sev, case=False, na=False).sum()
                    feat[f'severity_{sev}_count'] = cnt
                    feat[f'severity_{sev}_pct']   = round(cnt / len(group) * 100, 2) if group.shape[0] else 0

            if 'severity_response_based' in group.columns:
                for rc in response_cats:
                    cnt        = (group['severity_response_based'] == rc).sum()
                    clean_rc   = rc.replace(' - ', '_').replace(' ', '_')
                    feat[f'response_{clean_rc}_count'] = cnt
                    feat[f'response_{clean_rc}_pct']   = round(cnt / len(group) * 100, 2) if group.shape[0] else 0

            if 'response_time_days' in group.columns:
                vrt = group['response_time_days'].dropna()
                feat['avg_response_time_days']    = round(vrt.mean(),   2) if not vrt.empty else None
                feat['median_response_time_days'] = round(vrt.median(), 2) if not vrt.empty else None
                feat['min_response_time_days']    = vrt.min() if not vrt.empty else None
                feat['max_response_time_days']    = vrt.max() if not vrt.empty else None

            if 'date_of_completion' in group.columns:
                dts        = pd.to_datetime(group['date_of_completion'], errors='coerce').dropna()
                feat['earliest_completion'] = dts.min() if not dts.empty else None
                feat['latest_completion']   = dts.max() if not dts.empty else None
                feat['date_range_days']     = (dts.max() - dts.min()).days if len(dts) > 1 else 0

            if 'source_sheet'    in group.columns:
                feat['source_sheets']         = ', '.join(group['source_sheet'].unique())
            if 'estimated_value' in group.columns:
                feat['total_estimated_value'] = group['estimated_value'].sum()

            enhanced_history.append(feat)

        return pd.DataFrame(enhanced_history)

    def split_large_dataframe(self, df, base_name):
        if len(df) <= self.config.EXCEL_ROW_LIMIT:
            return {base_name: df}
        chunks    = {}
        n_chunks  = len(df) // self.config.EXCEL_ROW_LIMIT + 1
        for i in range(n_chunks):
            s   = i * self.config.EXCEL_ROW_LIMIT
            e   = min((i + 1) * self.config.EXCEL_ROW_LIMIT, len(df))
            key = f"{base_name}_part{i+1}"[:self.config.EXCEL_SHEET_NAME_LIMIT]
            chunks[key] = df.iloc[s:e].copy()
        return chunks

    def generate_stats(self, all_dfs):
        print("\n🔄 Generating combined statistics...")
        if not self.feature_history or not all_dfs:
            print("  ⚠️  Insufficient data for statistics.")
            return

        combined = pd.concat(all_dfs, ignore_index=True)

        print("  📊 Building enhanced feature work history...")
        enhanced = self.build_enhanced_feature_history(combined)
        enhanced = enhanced.sort_values('total_activities', ascending=False)

        feature_by_prefix = {}
        for prefix in enhanced['feature_prefix'].unique():
            pdata = enhanced[enhanced['feature_prefix'] == prefix].copy()
            for dc in ['earliest_completion', 'latest_completion']:
                if dc in pdata.columns:
                    pdata[dc] = pdata[dc].apply(
                        lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A')
            feature_by_prefix[prefix] = pdata

        out = os.path.join(self.config.OUTPUT_DIR, 'combined_statistics.xlsx')
        print("  💾 Writing Excel file...")

        with pd.ExcelWriter(out, engine='openpyxl') as w:
            pd.DataFrame([{
                'Total_Original_Records': len(self.feature_history),
                'Total_Activities':       len(combined),
                'Unique_Features':        combined['feature_number'].nunique(),
            }]).to_excel(w, sheet_name='Overall_Summary', index=False)

            cat = combined['primary_category'].value_counts().reset_index()
            cat.columns = ['Category', 'Count']
            cat['Percentage'] = (cat['Count'] / len(combined) * 100).round(1)
            for sn, chunk in self.split_large_dataframe(cat, 'Category_Distribution').items():
                chunk.to_excel(w, sheet_name=sn, index=False)

            if 'source_sheet' in combined.columns:
                cs = pd.crosstab(combined['source_sheet'],
                                 combined['primary_category']).reset_index()
                for sn, chunk in self.split_large_dataframe(cs, 'Category_By_Sheet').items():
                    chunk.to_excel(w, sheet_name=sn, index=False)

            if 'response_time_days' in combined.columns:
                vr = combined[combined['response_time_days'].notna()]
                if not vr.empty:
                    resp = (vr.groupby('severity_response_based')['response_time_days']
                            .agg(['count', 'mean', 'median', 'min', 'max']).round(2).reset_index())
                    resp.to_excel(w, sheet_name='Response_Time_Analysis', index=False)

            for prefix in sorted(feature_by_prefix.keys()):
                pdata = feature_by_prefix[prefix]
                sn    = f'Feature_History_{prefix}'[:self.config.EXCEL_SHEET_NAME_LIMIT]
                for sheet_n, chunk in self.split_large_dataframe(pdata, sn).items():
                    chunk.to_excel(w, sheet_name=sheet_n, index=False)

            if 'feature_prefix' in enhanced.columns:
                ps = (enhanced.groupby('feature_prefix')
                      .agg(Unique_Features=('total_activities', 'count'),
                           Total_Activities=('total_activities', 'sum'),
                           Avg_Activities_Per_Feature=('total_activities', 'mean'))
                      .round(2).reset_index())
                ps.to_excel(w, sheet_name='Prefix_Summary', index=False)

            if 'completion_year' in combined.columns:
                yr = (combined.groupby('completion_year')
                      .agg(Activity_Count=('activity_id', 'count'),
                           Top_Category=('primary_category',
                                         lambda x: x.mode()[0] if not x.mode().empty else 'N/A'))
                      .reset_index())
                yr.to_excel(w, sheet_name='Yearly_Analysis', index=False)

        print(f"  ✅ Combined stats saved: {out}")

    def run(self):
        print("=" * 70)
        print("MAINTENANCE CLASSIFICATION PIPELINE")
        print(f"Input:   {self.config.INPUT_FILE}")
        print(f"Output:  {self.config.OUTPUT_DIR}")
        print(f"Workers: {self.config.MAX_WORKERS}")
        print("=" * 70)

        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        all_dfs = []

        for sheet in self.config.SHEETS_TO_PROCESS:
            try:
                df = pd.read_excel(self.config.INPUT_FILE, sheet_name=sheet)
                self.process_sheet(sheet, df)
                csv_path = os.path.join(self.config.OUTPUT_DIR, 'by_sheet',
                                        f'{sheet}_activities.csv')
                if os.path.exists(csv_path):
                    all_dfs.append(pd.read_csv(csv_path))
            except Exception as e:
                print(f"  ❌ Error processing {sheet}: {e}")

        if all_dfs or self.feature_history:
            self.generate_stats(all_dfs)

        print(f"\n{'='*70}\nPROCESSING COMPLETE\n{'='*70}")
        for sheet, stats in self.progress.sheets.items():
            print(f"  {sheet}: {stats['done']:,} rows in "
                  f"{(time.time()-stats['start'])/60:.1f}m")
        print(f"\n📁 Output: {self.config.OUTPUT_DIR}")


# ============================================
# MAIN
# ============================================

def main():
    Paths.ensure_output_dirs()
    config = Config()

    # Clear checkpoints to force full reprocessing
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"Cleared checkpoints: {checkpoint_dir}")

    processor = SewerProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
