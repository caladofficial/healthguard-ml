"""Shared config for the curation + fine-tuning pipeline."""
import os, json, hashlib, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, 'data', 'pipeline')
REPORTS = os.path.join(ROOT, 'reports', 'pipeline')
for d in (PIPE, REPORTS):
    os.makedirs(d, exist_ok=True)

SEED = 42
TARGET_RAW = 300_000          # user requirement: >= 300k raw examples before cleaning
SYNTH_SHARDS = [(100_000, 42), (100_000, 43), (100_000, 44)]   # 3 x 100k "collection sites"

# The 45-column schema the shipped web model expects (from triage_synth).
FEATURES = ['age','sex','pregnancy','hr','sbp','dbp','rr','temp_c','spo2','gcs','pain',
            'onset_hours','followup_flag','shock_index','pulse_pressure','temp_deviation',
            'spo2_gap','qsofa_feat','sirs_feat','news_like','age_risk','p_dm','p_cad',
            'p_stroke','p_sepsis','p_liver','p_maternal'] + [
            'chest_pain','dyspnea','abd_pain','fever','cough','headache','weakness',
            'bleeding','vomiting','dizziness','syncope','seizure','altered_mental',
            'palpitations','jaundice','rash','htn','dm','cad_hf','copd','ckd','cancer',
            'stroke_hx','immuno']
LABEL = 'triage'
CLASSES = ['T0', 'T1', 'T2', 'T3', 'T4']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def row_hash(df, cols):
    """Vectorised content hash per record — for exact-duplicate detection.

    pd.util.hash_pandas_object is ~100x faster than a per-row string join;
    the string version is what stalled the first run (300k x 51 conversions).
    """
    import pandas as pd
    h = pd.util.hash_pandas_object(df[cols].fillna(-999).astype('float32'), index=False)
    return h.astype('uint64').astype(str).str.zfill(20)


def stamp():
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def log_json(name, obj):
    path = os.path.join(REPORTS, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    return path


P_DROP = ['p_dm', 'p_cad', 'p_stroke', 'p_sepsis', 'p_liver', 'p_maternal']
BARE = [c for c in FEATURES if c not in P_DROP]


def prep(df, cols=None):
    """Single preprocessing entry point for every stage.

    Stage 5 passed NaNs through (LightGBM splits on them natively) while stage 6
    median-filled first, so the two stages scored the same checkpoint differently
    (0.9221 vs 0.9242). All stages now go through here.
    """
    cols = cols or BARE
    return df[cols].astype('float32')


def prep_imputed(df, med=None, cols=None):
    """Imputed copy of prep(), for estimators that cannot take NaN (PCA, scalers).

    Model inference uses prep() so scoring matches stage 5 exactly; only the
    error-clustering geometry needs the imputed version.
    """
    X = prep(df, cols)
    med = X.median() if med is None else med
    return X.fillna(med)
