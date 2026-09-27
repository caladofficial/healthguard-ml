"""collect2 — EXPANDED CLINICAL DATA COLLECTION (round 2, accuracy push).
EVOLVEX IT SOLUTIONS PVT. LTD.

New public sources (UCI API data.csv direct + OpenML):
  UCI 296  Diabetes 130-US Hospitals (≈101k)  → diabetes_readmit head
  UCI 519  Heart Failure Clinical Records     → hf_mortality head
  UCI 529  Early Stage Diabetes Risk          → diabetes_symptoms head
  UCI 571  HCV blood labs                     → liver_hcv head
  OpenML 337 SPECTF (cardiac perfusion)       → cad_spect head
BUPA (UCI 60) deliberately SKIPPED: the classic 'selector' label has disputed
semantics (split marker vs outcome) — documented here rather than used.

Appends to data/raw/MANIFEST.json (spec §13).
"""
import json, os, sys, io
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

RAW = C.RAW

UCI = [
    (296, 'diabetes_130', 'Diabetes 130-US Hospitals for Years 1999-2008',
     'https://archive.ics.uci.edu/static/public/296/data.csv',
     'CC BY 4.0', 'UCI ML Repository id 296 (Strack et al. 2014)'),
    (519, 'heart_failure', 'Heart Failure Clinical Records',
     'https://archive.ics.uci.edu/static/public/519/data.csv',
     'CC BY 4.0', 'UCI ML Repository id 519 (Chicco & Jurman 2020)'),
    (529, 'early_diabetes', 'Early Stage Diabetes Risk Prediction',
     'https://archive.ics.uci.edu/static/public/529/data.csv',
     'CC BY 4.0', 'UCI ML Repository id 529 (Islam et al. 2020)'),
    (571, 'hcv_liver', 'HCV data (blood donors vs hepatitis C)',
     'https://archive.ics.uci.edu/static/public/571/data.csv',
     'CC BY 4.0', 'UCI ML Repository id 571'),
]


def main():
    man_path = os.path.join(RAW, 'MANIFEST.json')
    man = json.load(open(man_path)) if os.path.exists(man_path) else []
    if isinstance(man, dict):        # collect.py writes dict form — normalize to list
        man = list(man.values())
    ok = 0

    for uid, slug, title, url, lic, prov in UCI:
        try:
            df = pd.read_csv(url)
            out = os.path.join(RAW, f'{slug}.csv')
            df.to_csv(out, index=False)
            man.append({'folder': slug, 'title': title, 'source': prov, 'url': url,
                        'license': lic, 'retrieved': '2026-09-24', 'synthetic': False,
                        'rows': int(df.shape[0]), 'cols': int(df.shape[1])})
            ok += 1
            print(f'✓ {slug:18s} {df.shape}  ← UCI {uid}')
        except Exception as e:
            print(f'✗ {slug:18s} UCI {uid}: {type(e).__name__}: {e}')

    # SPECTF via OpenML (fetch_openml handles ARFF; as_frame keeps names)
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=337, as_frame=True, parser='auto')
        df = d.data.copy()
        df['target'] = d.target.astype(str)
        out = os.path.join(RAW, 'spectf.csv')
        df.to_csv(out, index=False)
        man.append({'folder': 'spectf', 'title': 'SPECTF cardiac perfusion (OpenML 337)',
                    'source': 'OpenML data id 337 (originally UCI SPECTF)',
                    'url': 'https://www.openml.org/d/337', 'license': 'UCI: unspecified/academic',
                    'retrieved': '2026-09-24', 'synthetic': False,
                    'rows': int(df.shape[0]), 'cols': int(df.shape[1])})
        ok += 1
        print(f'✓ spectf             {df.shape}  ← OpenML 337')
    except Exception as e:
        print(f'✗ spectf             OpenML 337: {type(e).__name__}: {e}')

    man.append({'folder': 'bupa_liver_skipped', 'title': 'BUPA Liver Disorders (UCI 60)',
                'decision': 'SKIPPED — selector label semantics disputed (split marker vs '
                            'outcome); HCV labs used for the liver panel instead'})
    json.dump(man, open(man_path, 'w'), indent=2)
    print(f'\ncollect2: {ok}/5 new sources → data/raw/ (manifest updated)')


if __name__ == '__main__':
    main()
