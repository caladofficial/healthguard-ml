"""02 — DATASET HARMONIZERS (round 2: 12 heads) → data/processed/*.csv
EVOLVEX IT SOLUTIONS PVT. LTD.

Round-2 roster (accuracy push):
  real:  cad(+statlog aux + SPECTF), diabetes(Pima + symptoms + readmit),
         fetal_risk(CTG), maternal_risk, liver(ILPD + HCV), hf_mortality
  synth: sepsis 30k (family E, engineered physiology)
Every set → data/processed/<name>.X.csv + <name>.y.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
warnings.filterwarnings('ignore')

RAW = C.RAW
PROC = C.PROC

MATERNAL_MAP = {'low risk': 0, 'mid risk': 1, 'high risk': 2}


def _folder(name):
    return os.path.join(RAW, name)


# --------------------------------------------------------------------------
def load_cad():
    frames = []
    for cohort in ['cleveland', 'hungarian', 'switzerland', 'va']:
        path = os.path.join(_folder('heart_disease'), f'processed.{cohort}.data')
        df = pd.read_csv(path, header=None)
        df.replace('?', np.nan, inplace=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    y = df.pop(58 if 58 in df.columns else df.columns[-1])
    keep = y.notna()
    df, y = df[keep], y[keep]
    # keep columns with <30% missing; median-impute the rest (clinical standard)
    df = df.dropna(axis=1, thresh=int(0.7 * len(df)))
    df = df.fillna(df.median())
    y = y.astype(float).map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1}).astype(int)
    df.columns = [f'f{i}' for i in range(df.shape[1])]
    return df.reset_index(drop=True), y.reset_index(drop=True), 'binary'


def load_diabetes():
    df = pd.read_csv(os.path.join(RAW, 'pima.csv'))
    y = (df.pop('__target__').astype(str).str.strip() == 'tested_positive').astype(int)
    return df, y, 'binary'


def load_fetal_risk():
    path = os.path.join(_folder('cardiotocography'), 'CTG.xls')
    sheets = pd.read_excel(path, sheet_name=None)
    # prefer the sheet that actually carries the risk target (ties on rows pick
    # the wrong 'Data' sheet with Unnamed columns otherwise)
    df = None
    for cand in sheets.values():
        cols = [str(c).strip().upper() for c in cand.columns]
        if 'NSP' in cols or 'CLASS' in cols:
            df = cand; break
    if df is None:
        df = max(sheets.values(), key=len)
    df.columns = [str(c).strip() for c in df.columns]
    key = next(c for c in df.columns if c.upper() == 'NSP')
    y = pd.to_numeric(df[key], errors='coerce') - 1
    X = df.drop(columns=[c for c in ['FileName', 'Date', 'SegFile', 'CLASS', key] if c in df.columns])
    X = X.select_dtypes('number')
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok].reset_index(drop=True), y[ok].astype(int).reset_index(drop=True)
    return X, y, 'multiclass'


def load_maternal_risk():
    df = pd.read_csv(os.path.join(_folder('maternal_health'), 'Maternal Health Risk Data Set.csv'),
                     encoding='utf-8-sig')
    y = df.pop('RiskLevel').astype(str).str.strip().str.lower().map(MATERNAL_MAP).astype(int)
    return df, y, 'multiclass'


def load_liver():
    path = os.path.join(_folder('ilpd'), 'Indian Liver Patient Dataset (ILPD).csv')
    df = pd.read_csv(path, header=None)
    df.columns = ['age', 'sex', 'tb', 'db', 'alkphos', 'sgpt', 'sgot', 'tp', 'alb', 'ag_ratio', 'label']
    df['sex'] = (df['sex'].astype(str).str.upper().str.startswith('M')).astype(int)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(df.median())   # ag_ratio has a few missing values
    y = (2 - df.pop('label').astype(int)).clip(0, 1)   # selector 1=patient → positive
    return df, y, 'binary'


def load_cad_aux():
    df = pd.read_csv(os.path.join(RAW, 'heart_statlog.csv'))
    y = (df.pop('__target__').astype(str).str.strip() == 'present').astype(int)
    return df, y, 'binary'


# ---------------- round-2 loaders ------------------------------------------
def load_diabetes_readmit():
    """UCI 296 — early (<30-day) readmission risk among diabetic inpatients."""
    df = pd.read_csv(os.path.join(RAW, 'diabetes_130.csv'), low_memory=False)
    y = (df['readmitted'].astype(str).str.strip() == '<30').astype(int)
    X = pd.DataFrame()
    X['age_mid'] = df['age'].astype(str).str.extract(r'\[(\d+)-')[0].astype(float) + 5
    X['time_in_hospital'] = df['time_in_hospital']
    X['num_lab_procedures'] = df['num_lab_procedures']
    X['num_procedures'] = df['num_procedures']
    X['num_medications'] = df['num_medications']
    X['number_outpatient'] = df['number_outpatient']
    X['number_emergency'] = df['number_emergency']
    X['number_inpatient'] = df['number_inpatient']
    X['number_diagnoses'] = df['number_diagnoses']
    X['n_prior_util'] = df[['number_outpatient', 'number_emergency', 'number_inpatient']].sum(axis=1)
    X['admission_type_id'] = df['admission_type_id']
    X['discharge_disposition_id'] = df['discharge_disposition_id']
    for col, mp in [('max_glu_serum', {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}),
                    ('A1Cresult', {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}),
                    ('insulin', {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}),
                    ('change', {'No': 0, 'Ch': 1}),
                    ('diabetesMed', {'No': 0, 'Yes': 1})]:
        X[col] = df[col].astype(str).str.strip().map(mp).fillna(0)
    for col in ['metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']:
        if col in df.columns:
            X[col] = (df[col].astype(str).str.strip() != 'No').astype(int)

    def icd_bucket(v):
        try:
            f = float(str(v).split('.')[0])
        except Exception:
            return 'other'
        if 250 <= f < 251: return 'diabetes'
        if 390 <= f <= 459 or 785 <= f < 786: return 'circulatory'
        if 460 <= f <= 519 or 786 <= f < 787: return 'respiratory'
        if 520 <= f <= 579 or 787 <= f < 788: return 'digestive'
        if 580 <= f <= 629: return 'genitourinary'
        if 800 <= f <= 999: return 'injury'
        if 710 <= f <= 739: return 'musculoskeletal'
        if 680 <= f <= 709: return 'skin'
        if 290 <= f <= 319: return 'mental'
        if 140 <= f <= 239: return 'neoplasm'
        return 'other'
    for i in (1, 2, 3):
        b = df[f'diag_{i}'].map(icd_bucket)
        for k in ['diabetes', 'circulatory', 'respiratory', 'digestive', 'genitourinary', 'injury', 'neoplasm']:
            X[f'diag{i}_{k}'] = (b == k).astype(int)
    for k in ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other']:
        X[f'race_{k}'] = (df['race'].astype(str).str.strip() == k).astype(int)
    X['female'] = (df['gender'].astype(str).str.strip().str.lower() == 'female').astype(int)
    return X, y, 'binary'


def load_hf_mortality():
    """UCI 519 — mortality during follow-up in heart-failure patients."""
    df = pd.read_csv(os.path.join(RAW, 'heart_failure.csv'))
    y = df.pop('death_event').astype(int)
    return df, y, 'binary'


def load_diabetes_symptoms():
    """UCI 529 — early-stage diabetes risk from signs/symptoms."""
    df = pd.read_csv(os.path.join(RAW, 'early_diabetes.csv'))
    y = (df.pop('class').astype(str).str.strip().str.lower() == 'positive').astype(int)
    X = pd.DataFrame()
    X['age'] = df.pop('age')
    X['male'] = (df.pop('gender').astype(str).str.strip().str.lower() == 'male').astype(int)
    for c in df.columns:
        X[c] = (df[c].astype(str).str.strip().str.lower() == 'yes').astype(int)
    return X, y, 'binary'


def load_liver_hcv():
    """UCI 571 — 3-class liver damage from blood labs (0 healthy, 1 disease, 2 cirrhosis)."""
    df = pd.read_csv(os.path.join(RAW, 'hcv_liver.csv'))
    y = df['Category'].astype(str).map({
        '0=Blood Donor': 0, '0s=suspect Blood Donor': 0,
        '1=Hepatitis': 1, '2=Fibrosis': 1, '3=Cirrhosis': 2}).astype(int)
    X = df.drop(columns=['ID', 'Category'])
    X['Sex'] = (X['Sex'].astype(str).str.strip().str.lower() == 'm').astype(int)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    ok = X.notna().all(axis=1)
    return X[ok].reset_index(drop=True), y[ok].reset_index(drop=True), 'multiclass'


def load_cad_spect():
    """OpenML 337 SPECTF — myocardial perfusion abnormality (CAD family)."""
    df = pd.read_csv(os.path.join(RAW, 'spectf.csv'))
    y = df.pop('target').astype(str)
    vals = sorted(y.unique())
    m = {vals[0]: 0, vals[-1]: 1}          # 1=normal → 0 ; 2=abnormal → 1
    y = y.map(m).astype(int)
    X = df.astype(float)
    return X, y, 'binary'


# ---------------- synthetic sepsis (family E) ------------------------------
def _make_sepsis(n=30000, seed=13):
    """qSOFA-structured sepsis physiology + engineered clinical features."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=(n, 6))
    hr = 70 + 55 * p[:, 0] + rng.normal(0, 10, n)
    sbp = 130 - 50 * p[:, 1] + rng.normal(0, 12, n)
    dbp = 80 - 22 * p[:, 1] + rng.normal(0, 8, n)
    rr = 13 + 11 * p[:, 2] + rng.normal(0, 2.8, n)
    temp = 36.4 + 1.8 * p[:, 3] + rng.normal(0, 0.4, n)
    spo2 = 99 - 8 * p[:, 4] - 3 * p[:, 5] + rng.normal(0, 1.4, n)
    X = pd.DataFrame({'hr': hr, 'sbp': sbp, 'dbp': dbp, 'rr': rr,
                      'temp_c': temp, 'spo2': spo2, 'gcs': 15 - np.round(2 * p[:, 5]).astype(int)})
    X['shock_index'] = hr / sbp
    X['pulse_pressure'] = sbp - dbp
    X['temp_deviation'] = np.abs(temp - 36.8)
    X['spo2_gap'] = np.clip(100 - spo2, 0, None)
    X['rr_excess'] = np.clip(rr - 18, 0, None)
    X['qsofa_est'] = ((sbp <= 100).astype(int) + (rr >= 22).astype(int) + (X['gcs'] <= 13).astype(int))
    sepsis = (p.sum(axis=1) > 3.05).astype(int)
    flip = rng.random(n) < 0.05
    sepsis ^= flip.astype(int)
    return X, sepsis, 'binary'


# --------------------------------------------------------------------------
def load_all():
    return {
        'cad': load_cad(),
        'diabetes': load_diabetes(),
        'fetal_risk': load_fetal_risk(),
        'maternal_risk': load_maternal_risk(),
        'liver': load_liver(),
        'sepsis': _make_sepsis(),
        'cad_aux': load_cad_aux(),
        'diabetes_readmit': load_diabetes_readmit(),
        'hf_mortality': load_hf_mortality(),
        'diabetes_symptoms': load_diabetes_symptoms(),
        'liver_hcv': load_liver_hcv(),
        'cad_spect': load_cad_spect(),
    }


def summarize():
    sets = load_all()
    for name, (X, y, task) in sets.items():
        X.to_csv(os.path.join(PROC, f'{name}.X.csv'), index=False)
        pd.Series(np.asarray(y), name='label').to_frame().to_csv(
            os.path.join(PROC, f'{name}.y.csv'), index=False)
        counts = np.bincount(np.asarray(y, dtype=int)).tolist()
        print(f"✓ {name:18s} X={X.shape} task={task} class_counts={counts}")
    print(f"\nsaved {len(sets)} harmonized sets → data/processed/")


if __name__ == '__main__':
    summarize()
