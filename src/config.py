"""HealthGuard Triage Risk Engine — global config.
EVOLVEX IT SOLUTIONS PVT. LTD.
"""
import os, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'raw')
PROC = os.path.join(ROOT, 'data', 'processed')
MODELS = os.path.join(ROOT, 'models')
REPORTS = os.path.join(ROOT, 'reports')
for d in (RAW, PROC, MODELS, REPORTS):
    os.makedirs(d, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED)

TRIAGE_CLASSES = ['T0', 'T1', 'T2', 'T3', 'T4']  # emergency → follow-up

# Model persistence names
SPECIALIST_TASKS = ['cad', 'diabetes', 'fetal_risk', 'maternal_risk', 'liver', 'sepsis',
                    'cad_aux', 'diabetes_readmit', 'hf_mortality', 'diabetes_symptoms',
                    'liver_hcv', 'cad_spect']
