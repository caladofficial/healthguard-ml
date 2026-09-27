"""Stage 1 — COLLECTION with provenance.

Rules we hold to (from the brief):
  * respect robots.txt / ToS, record the licence of every source;
  * skip anything gated behind login or without a compatible licence
    (MIMIC-IV-ED full extract needs PhysioNet credentialing -> skipped, and the
    attempt is logged rather than silently dropped);
  * strip personal/identifying information (none of these sources carry direct
    identifiers, and we keep no id columns);
  * every record carries provenance: source id, URL, retrieval date, licence,
    content hash.

Output: data/pipeline/raw.parquet  +  reports/pipeline/01_collection.{json,md}
"""
import os, sys, json, hashlib, datetime, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import pipeline_cfg as C          # noqa: E402
import triage_synth               # noqa: E402

RAW = os.path.join(ROOT, 'data')
PIPE = C.PIPE
PROV = []          # provenance + audit records


def note(record):
    PROV.append(record)
    flag = {'admitted': 'ADMITTED', 'rejected': 'REJECTED', 'failed': 'UNAVAILABLE'}[record['status']]
    print(f"  {flag:11} {record['source_id']:24} rows={record.get('rows_admitted', 0):>7}  {record['notes'][:52]}")


def robots_status(url):
    """Fetch robots.txt for the host; report what it says (not just that we fetched it)."""
    try:
        host = '/'.join(url.split('/')[:3])
        out = subprocess.run(['curl', '-s', '--max-time', '20', host + '/robots.txt'],
                             capture_output=True, text=True).stdout
        if out.lstrip().startswith('<!DOCTYPE') or out.lstrip().startswith('<html'):
            return 'none_published (200 HTML page, no directives)'
        disallow = [l for l in out.splitlines() if l.lower().startswith('disallow')]
        return 'fetched, %d disallow rules' % len(disallow)
    except Exception as e:
        return 'fetch failed: %s' % str(e)[:40]


# --------------------------------------------------------------------- 1. synthetic
def collect_synthetic():
    """Three independent 'collection sites' (different seeds) -> realistic
    multi-site redundancy in the raw corpus."""
    frames = []
    for k, (n, seed) in enumerate(C.SYNTH_SHARDS):
        sid = f'synth_site_{"abc"[k]}'
        X, y, meta = triage_synth.generate(n=n, seed=seed)
        d = X.copy()
        del X
        d[C.LABEL] = y.astype(int)
        d['syndrome'] = meta['syndrome'].astype('category').values
        d['reason_codes'] = meta['reason_codes'].astype('category').values
        d['uncertain'] = meta['uncertain'].values
        d['label_source'] = 'protocol'
        d['source_id'] = sid
        # float32 halves memory on a 2 GB box
        for c in d.columns:
            if d[c].dtype == 'float64':
                d[c] = d[c].astype('float32')
        d['row_hash'] = C.row_hash(d, [c for c in C.FEATURES if c in d.columns])
        d.to_parquet(os.path.join(PIPE, f'raw_{sid}.parquet'), index=False)
        print(f'    {sid}: {len(d):,} rows written', flush=True)
        frames.append(d[['row_hash']])   # keep only the key; reload shards at the end
        note({
            'source_id': sid,
            'source_name': f'HealthGuard synthetic ED cohort (site {"ABC"[k]})',
            'source_url': 'local:src/triage_synth.py',
            'retrieval_date': C.stamp(),
            'license': 'self-generated synthetic; no third-party rights',
            'robots': 'n/a (locally generated)',
            'sha256': hashlib.sha256(
                (f'triage_synth:generate:n={n}:seed={seed}').encode()).hexdigest()[:32],
            'rows_in': n, 'rows_admitted': n, 'status': 'admitted',
            'notes': 'deterministic ESI/ATS-shaped protocol labelling, 5% borderline noise',
        })
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------- 2. real open data (harmonised)
# Columns each UCI set can honestly contribute to the triage schema.
UCI_MAPS = {
    'uci_863': {'name': 'Maternal Health Risk (UCI 863)',
                'license': 'CC BY 4.0',
                'url': 'https://archive.ics.uci.edu/dataset/863/maternal+health+risk',
                'map': {'Age': 'age', 'SystolicBP': 'sbp', 'DiastolicBP': 'dbp',
                        'BodyTemp': 'temp_c', 'HeartRate': 'hr'},
                'sex_from': None},
    'uci_45': {'name': 'Heart Disease (UCI 45)',
               'license': 'CC BY 4.0',
               'url': 'https://archive.ics.uci.edu/dataset/45/heart+disease',
               'map': {'age': 'age', 'sex': 'sex', 'trestbps': 'sbp', 'thalach': 'hr'},
               'sex_from': None},
}

# Sources we deliberately do NOT admit, with the reason recorded.
REJECTS = [
    ('uci_296', 'Diabetes 130-US hospitals (UCI 296)', 101766,
     'no vital signs - demographics/utilisation only, cannot populate triage schema'),
    ('uci_519', 'Heart failure clinical records (UCI 519)', 299,
     'no vital signs - labs + echocardiography only'),
    ('uci_529', 'Diabetes symptoms (UCI 529)', 520, 'no vital signs - symptom questionnaire'),
    ('uci_571', 'Hepatitis C (UCI 571)', 615, 'no vital signs - laboratory panel'),
    ('uci_827', 'Sepsis survival minimal clinical records (UCI 827)', 110341,
     'fetched live: only age/sex/episode/outcome, no vitals despite the name'),
]


def rule_label_from_vitals(d):
    """Apply the SAME deterministic escalation thresholds the web safety layer uses,
    but tolerating missing columns. Used only for partially-featured real records.

    T0 critical -> T1 major -> T2 priority -> T3 routine.
    """
    def col(name, default=np.nan):
        return d[name] if name in d.columns else pd.Series(default, index=d.index)

    sbp, hr, spo2, gcs, temp, rr, pain = (col(x) for x in
                                          ('sbp', 'hr', 'spo2', 'gcs', 'temp_c', 'rr', 'pain'))
    y = pd.Series(3, index=d.index)
    crit = (sbp < 90) | (hr < 40) | (hr > 150) | (spo2 < 88) | (gcs <= 8)
    y[crit.fillna(False)] = 0
    maj = (spo2 < 92) | (gcs <= 12) | (pain >= 8) | (temp > 39.5) | (rr >= 22)
    y[maj.fillna(False) & (y == 3)] = 1
    pri = (pain >= 5) | (temp > 38.2) | (hr > 105) | (sbp > 160)
    y[pri.fillna(False) & (y == 3)] = 2
    return y.astype(int)


def collect_real_local():
    frames = []
    for sid, spec in UCI_MAPS.items():
        path = os.path.join(RAW, f'{sid}.csv')
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path)
        d = pd.DataFrame(index=raw.index)
        for src, dst in spec['map'].items():
            if src in raw.columns:
                d[dst] = pd.to_numeric(raw[src], errors='coerce')
        # maternal set: every patient is pregnant by construction
        if sid == 'uci_863':
            d['pregnancy'] = 1
            d['sex'] = 1
        mapped = [c for c in d.columns if c in C.FEATURES]
        if len(set(spec['map'].values()) & set(C.FEATURES)) == 0:
            continue
        d[C.LABEL] = rule_label_from_vitals(d)
        d['label_source'] = 'rule_from_partial_vitals'
        d['source_id'] = sid
        d['syndrome'] = 'real_proxy'
        d['reason_codes'] = 'RULE_FROM_PARTIAL_VITALS'
        d['uncertain'] = 1          # partial features -> treat as uncertain
        frames.append(d)
        note({
            'source_id': sid, 'source_name': spec['name'], 'source_url': spec['url'],
            'retrieval_date': C.stamp(), 'license': spec['license'],
            'robots': robots_status(spec['url']),
            'sha256': C.sha256_file(path),
            'rows_in': int(len(raw)), 'rows_admitted': int(len(d)), 'status': 'admitted',
            'notes': f'harmonised {len(mapped)}/{len(C.FEATURES)} triage columns; '
                     f'rest missing by construction',
        })
    return frames


def record_rejections():
    for sid, name, rows, why in REJECTS:
        note({
            'source_id': sid, 'source_name': name, 'source_url':
                f'https://archive.ics.uci.edu/dataset/{sid.split("_")[1]}',
            'retrieval_date': C.stamp(), 'license': 'n/a - not used',
            'robots': robots_status('https://archive.ics.uci.edu/'),
            'rows_in': rows, 'rows_admitted': 0, 'status': 'rejected', 'notes': why,
        })


def probe_gated_sources():
    """Attempt the credentialed / unreachable real ED sources so the audit shows
    they were tried, not overlooked."""
    attempts = [
        ('mimic_iv_ed', 'MIMIC-IV-ED (PhysioNet)', 'https://physionet.org/content/mimic-iv-ed/2.2/',
         'REAL ED triage vitals + acuity - but full extract is credentialed access'),
        ('mimic_iv_ed_demo', 'MIMIC-IV-ED demo (PhysioNet, open)',
         'https://physionet.org/files/mimic-iv-ed-demo/2.2/edstays.csv',
         'open-access demo - file paths returned 404 in this environment'),
        ('nhamcs', 'NHAMCS ED public-use files (CDC)',
         'https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/',
         'reachable, but fixed-width layout needs a multi-MB SAS spec per year'),
        ('kaggle', 'Kaggle datasets', 'https://www.kaggle.com/api/v1',
         'kaggle.json holds a username-only key - API authentication unavailable'),
    ]
    for sid, name, url, why in attempts:
        code = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                               '--max-time', '20', url], capture_output=True, text=True).stdout
        note({
            'source_id': sid, 'source_name': name, 'source_url': url,
            'retrieval_date': C.stamp(), 'license': 'unknown - not ingested',
            'robots': robots_status(url), 'http_status': code,
            'rows_in': 0, 'rows_admitted': 0, 'status': 'failed', 'notes': why,
        })


def main():
    print('Stage 1 — collection')
    synth = collect_synthetic()
    real_frames = collect_real_local()
    record_rejections()
    probe_gated_sources()

    shards = [pd.read_parquet(os.path.join(PIPE, f'raw_synth_site_{k}.parquet'))
              for k in 'abc'] + real_frames
    df = pd.concat(shards, ignore_index=True)
    del shards
    # keep only schema columns that exist anywhere, in a stable order
    keep = [c for c in C.FEATURES if c in df.columns]
    df['feature_completeness'] = df[keep].notna().mean(axis=1).round(3)

    out = os.path.join(PIPE, 'raw.parquet')
    df.to_parquet(out, index=False)

    admitted = int(sum(r.get('rows_admitted', 0) for r in PROV))
    summary = {
        'stage': '01_collect',
        'generated': C.stamp(),
        'raw_corpus_rows': int(len(df)),
        'target_raw': C.TARGET_RAW,
        'meets_target': bool(len(df) >= C.TARGET_RAW),
        'rows_admitted_from_sources': admitted,
        'rows_considered_total': int(sum(r.get('rows_in', 0) or 0 for r in PROV)),
        'sources_admitted': [r['source_id'] for r in PROV if r['status'] == 'admitted'],
        'sources_rejected': [r['source_id'] for r in PROV if r['status'] == 'rejected'],
        'sources_unavailable': [r['source_id'] for r in PROV if r['status'] == 'failed'],
        'class_counts': {C.CLASSES[k]: int(v) for k, v in
                         enumerate(df[C.LABEL].value_counts().sort_index())},
        'label_source_counts': df['label_source'].value_counts().to_dict(),
        'pii': 'no direct identifiers ingested; no id/name/contact columns retained',
        'provenance': PROV,
    }
    C.log_json('01_collection.json', summary)

    lines = ['# Stage 1 — collection', '',
             f"Raw corpus: **{len(df):,} rows** (target {C.TARGET_RAW:,} → "
             f"{'MET' if summary['meets_target'] else 'MISS'})", '',
             '| source | status | rows in | admitted | licence | notes |',
             '|---|---|---|---|---|---|']
    for r in PROV:
        lines.append(f"| {r['source_id']} | {r['status']} | {r.get('rows_in', 0):,} | "
                     f"{r.get('rows_admitted', 0):,} | {r.get('license', '')[:28]} | {r['notes'][:46]} |")
    lines += ['', 'Class balance: ' + json.dumps(summary['class_counts']), '',
              'Label provenance: ' + json.dumps(summary['label_source_counts'])]
    open(os.path.join(C.REPORTS, '01_collection.md'), 'w').write('\n'.join(lines) + '\n')
    print(f"\nraw corpus {len(df):,} rows -> {out}")
    print('report ->', os.path.join(C.REPORTS, '01_collection.md'))


if __name__ == '__main__':
    main()
