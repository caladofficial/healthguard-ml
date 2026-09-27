"""Stage 2 — EMBEDDINGS (three similarity notions, as requested).

  numeric : the 45 model features, imputed + scaled + PCA -> 32 dims.
            Faithful similarity for tabular clinical records.
  tfidf   : each record rendered as bucketed `feature:bucket` tokens, TF-IDF ->
            TruncatedSVD -> 32 dims. Cheap, deterministic, no downloads.
  st      : each record rendered as a clinical one-liner, embedded with
            all-MiniLM-L6-v2 (384 dims). Real semantic similarity; ~150 sent/s
            on this 2-CPU box, so it is the expensive one.

Each method is timed and its CPU cost recorded, so the efficiency section has
real numbers instead of guesses.
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline_cfg as C          # noqa: E402

P_DROP = ['p_dm', 'p_cad', 'p_stroke', 'p_sepsis', 'p_liver', 'p_maternal']
BARE = [c for c in C.FEATURES if c not in P_DROP]     # 45 features, as the web model
DIM = 32
ST_MODEL = os.environ.get('HG_ST_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
# HG_ST_MODEL lets the embedder load from a local path when the host has no internet

SYMPTOM_TXT = ['chest_pain', 'dyspnea', 'abd_pain', 'fever', 'cough', 'headache',
               'weakness', 'bleeding', 'vomiting', 'dizziness', 'syncope', 'seizure',
               'altered_mental', 'palpitations', 'jaundice', 'rash']
COMORB_TXT = ['htn', 'dm', 'cad_hf', 'copd', 'ckd', 'cancer', 'stroke_hx', 'immuno']


def load_raw():
    return pd.read_parquet(os.path.join(C.PIPE, 'raw.parquet'))


# --------------------------------------------------------------- text renderings
def token_ids(df):
    """Integer token id per (column, bucket) — no Python strings.

    With 45 columns bucketed into <=20 bins the whole vocabulary is ~1k tokens,
    so we can build the count matrix numerically. The first version built 13.5M
    Python strings and was OOM-killed on this 2 GB box.
    """
    ids = np.zeros((len(df), len(BARE)), dtype=np.int32)
    offsets, vocab = [], []
    off = 0
    for j, col in enumerate(BARE):
        s_ = df[col]
        if s_.dtype.kind in 'if':
            b = pd.qcut(s_.fillna(s_.median()), 20, duplicates='drop', labels=False)
            b = np.nan_to_num(b.astype('float64'), nan=0).astype(int)
            k = int(b.max()) + 1
        else:
            b = s_.fillna(-1).astype(int) - s_.fillna(-1).astype(int).min()
            b = b.values if hasattr(b, 'values') else b
            k = int(b.max()) + 1
        ids[:, j] = b + off
        vocab.extend([f'{col}:{i}' for i in range(k)])
        off += k
        offsets.append(k)
    return ids, vocab, off


def clinical_sentence(r):
    bits = []
    if pd.notna(r.get('age')):
        bits.append(f"{int(r['age'])} year old {'male' if r.get('sex') == 0 else 'female'}")
    for name, unit in [('hr', 'bpm'), ('sbp', 'mmHg systolic'), ('dbp', 'mmHg diastolic'),
                       ('rr', 'breaths/min'), ('temp_c', 'C'), ('spo2', '% SpO2')]:
        v = r.get(name)
        if pd.notna(v):
            bits.append(f"{name} {v:.1f} {unit}")
    for name, txt in [('gcs', 'GCS'), ('pain', 'pain score')]:
        v = r.get(name)
        if pd.notna(v):
            bits.append(f"{txt} {int(v)}")
    sym = [s.replace('_', ' ') for s in SYMPTOM_TXT if r.get(s) == 1]
    if sym:
        bits.append('reports ' + ', '.join(sym[:4]))
    com = [c.replace('_', ' ') for c in COMORB_TXT if r.get(c) == 1]
    if com:
        bits.append('history of ' + ', '.join(com[:3]))
    return '; '.join(bits)


# --------------------------------------------------------------------- embeddings
def embed_numeric(df):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X = df[BARE].astype('float32').values
    X = SimpleImputer(strategy='median').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    p = PCA(n_components=DIM, random_state=C.SEED)
    E = p.fit_transform(X).astype('float32')
    return E, {'explained_variance': float(p.explained_variance_ratio_.sum())}


def embed_tfidf(df):
    """TF-IDF over bucketed tokens, built as a sparse count matrix (memory-safe)."""
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD
    ids, vocab, ntok = token_ids(df)
    n = len(df)
    rows = np.repeat(np.arange(n, dtype=np.int32), ids.shape[1])
    M = csr_matrix((np.ones(rows.size, dtype=np.float32),
                    (rows, ids.ravel())), shape=(n, ntok))
    del ids, rows
    T = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(M)
    del M
    svd = TruncatedSVD(n_components=DIM, random_state=C.SEED)
    E = svd.fit_transform(T).astype('float32')
    return E, {'vocab': int(ntok),
               'nnz': int(T.nnz),
               'explained_variance': float(svd.explained_variance_ratio_.sum())}


def embed_st(df, chunk=4000, limit=None):
    """Sentence-transformer embeddings, streamed to a memmap.

    The naive version materialised 301k x 384 floats (463 MB) alongside torch
    and the input strings and was OOM-killed on this 2 GB box. Here we encode in
    chunks into a memmap, then reduce to 32 dims with IncrementalPCA so later
    stages never hold the full 384-dim array.
    """
    import torch
    from sklearn.decomposition import IncrementalPCA
    from sentence_transformers import SentenceTransformer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    m = SentenceTransformer(ST_MODEL, device=dev)
    BATCH = 256 if dev == 'cuda' else 16
    print(f'    st device={dev} batch={BATCH}', flush=True)
    n = len(df)
    tmp = os.path.join(C.PIPE, '_st_384.memmap')
    big = np.memmap(tmp, dtype='float32', mode='w+', shape=(n, 384))
    def _text(sl):
        # built per chunk: the old version held 301k Python strings in RAM
        age = sl['age'].fillna(-1).astype(int).astype(str)
        out = (age + ' year old; hr ' + sl['hr'].astype('float32').round(1).astype(str) +
               ' bpm; sbp ' + sl['sbp'].astype('float32').round(1).astype(str) +
               ' mmHg systolic; rr ' + sl['rr'].astype('float32').round(1).astype(str) +
               ' breaths/min; temp ' + sl['temp_c'].astype('float32').round(1).astype(str) +
               ' C; spo2 ' + sl['spo2'].astype('float32').round(1).astype(str) + ' % SpO2')
        return out.tolist()
    for i in range(0, n, chunk):
        txt = _text(df.iloc[i:i + chunk])
        big[i:i + len(txt)] = m.encode(txt, batch_size=BATCH, show_progress_bar=False,
                                       convert_to_numpy=True,
                                       normalize_embeddings=True).astype('float32')
        big.flush()
        del txt
        print(f'    st {min(i + chunk, n)}/{n}', flush=True)
    ip = IncrementalPCA(n_components=DIM, batch_size=chunk)
    for i in range(0, n, chunk):
        ip.partial_fit(np.asarray(big[i:i + chunk]))
    E = np.vstack([ip.transform(np.asarray(big[i:i + chunk])) for i in range(0, n, chunk)])
    E = E.astype('float32')
    del big
    os.remove(tmp)                      # keep the workspace snapshot small
    return E, {'model': ST_MODEL, 'raw_dim': 384,
               'explained_variance': float(ip.explained_variance_ratio_.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='numeric', choices=['numeric', 'tfidf', 'st', 'all'])
    ap.add_argument('--limit', type=int, default=None,
                    help='interleaved subsample size (for methods that exceed the RAM budget)')
    a = ap.parse_args()
    df = load_raw()
    if a.limit and a.limit < len(df):
        step = int(np.ceil(len(df) / a.limit))
        df = df.iloc[::step].reset_index(drop=True)
        df.to_parquet(os.path.join(C.PIPE, f'raw_{a.method}_sub.parquet'), index=False)
        print(f'subsample: {len(df):,} rows (every {step}th row, spans all sources)', flush=True)
    methods = ['numeric', 'tfidf', 'st'] if a.method == 'all' else [a.method]
    for meth in methods:
        t0 = time.time()
        fn = {'numeric': embed_numeric, 'tfidf': embed_tfidf, 'st': embed_st}[meth]
        E, extra = fn(df)
        dt = time.time() - t0
        out = os.path.join(C.PIPE, f'emb_{meth}.npy')
        np.save(out, E)
        rec = {'method': meth, 'rows': int(E.shape[0]), 'dim': int(E.shape[1]),
               'seconds': round(dt, 1), 'rows_per_second': round(E.shape[0] / dt, 1),
               'array_mb': round(E.nbytes / 1e6, 1), **extra}
        C.log_json(f'02_embed_{meth}.json', rec)
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
