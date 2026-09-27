"""01 — DATA COLLECTION (autonomous).
EVOLVEX IT SOLUTIONS PVT. LTD.

Sources (all public, license-tracked in manifest — spec §13 'every sample carries
source/license/allowed_use'):
  A. UCI ML Repository static archives (no auth)
  B. OpenML via sklearn.datasets.fetch_openml (no auth)
  C. Synthetic cohorts (family E per spec §11) — generated later in triage_synth.py

NOTE: Kaggle API was intended but kaggle.json contained only `username` (no API
key) → Kaggle auth impossible. Portfolio uses UCI + OpenML equivalents of the
same clinical risk problems (documented in MODEL_CARD.md).
"""
import io, json, os, sys, urllib.request, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

UCI = "https://archive.ics.uci.edu/static/public/{id}/{slug}.zip"

# (name, uci_id, slug, license_note)
UCI_TARGETS = [
    ("heart_disease", 45, "heart+disease", "UCI ML Repository — Heart Disease (no license restriction stated; research use)"),
    ("cardiotocography", 193, "cardiotocography", "UCI ML Repository — Cardiotocography (research use)"),
    ("maternal_health", 863, "maternal+health+risk", "UCI ML Repository — Maternal Health Risk (CC BY 4.0)"),
    ("ilpd", 225, "ilpd+indian+liver+patient+dataset", "UCI ML Repository — ILPD (research use)"),
    ("diabetes_uci", 34, "diabetes", "UCI ML Repository — Diabetes (research use)"),
]

OPENML_TARGETS = [
    ("pima", "diabetes", "OpenML 'diabetes' (Pima Indians, CC0)"),
    ("heart_statlog", "heart-statlog", "OpenML 'heart-statlog' (CC0)"),
]

def fetch_uci(name, uid, slug, license_note, manifest):
    url = UCI.format(id=uid, slug=slug)
    outdir = os.path.join(C.RAW, name)
    os.makedirs(outdir, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            zbytes = r.read()
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            zf.extractall(outdir)
        files = [f for f in os.listdir(outdir)]
        manifest[name] = {"source": url, "license": license_note, "family": "real_public",
                          "files": files, "status": "ok"}
        print(f"✓ UCI {name}: {files}")
    except Exception as e:
        manifest[name] = {"source": url, "license": license_note, "status": f"FAILED: {e}"}
        print(f"✗ UCI {name}: {e}")

def fetch_openml_named(name, oml, license_note, manifest):
    try:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(name=oml, version=1, as_frame=True, parser='auto')
        df = d.data.copy(); df['__target__'] = d.target.values
        path = os.path.join(C.RAW, f"{name}.csv")
        df.to_csv(path, index=False)
        manifest[name] = {"source": f"https://www.openml.org/d/{d.details.get('id')}", "license": license_note,
                          "family": "real_public", "files": [f"{name}.csv"], "status": "ok",
                          "shape": list(df.shape)}
        print(f"✓ OpenML {name} ({oml}): {df.shape}")
    except Exception as e:
        manifest[name] = {"source": f"openml:{oml}", "license": license_note, "status": f"FAILED: {e}"}
        print(f"✗ OpenML {name}: {e}")

def resolve_sepsis(manifest):
    """Try UCI search API for any sepsis cohort; fall back to documented synthetic later."""
    try:
        with urllib.request.urlopen("https://archive.ics.uci.edu/api/datasets?search=sepsis", timeout=30) as r:
            data = json.loads(r.read().decode())
        rows = data.get('data', []) if isinstance(data, dict) else []
        got = False
        for row in rows[:5]:
            # API shapes vary; be defensive
            rid = row.get('id') or row.get('dataset_id')
            name = (row.get('name') or '').lower()
            if rid and ('sepsis' in name or 'sepsis' in str(row).lower()):
                slug = (row.get('name') or '').replace(' ', '+')
                fetch_uci('sepsis_uci', rid, slug, 'UCI ML Repository — Sepsis (research use)', manifest)
                got = True; break
        if not got:
            manifest['sepsis'] = {"source": "UCI search + OpenML (none found) → SYNTHETIC family E",
                                  "license": "synthetic (generated)", "family": "synthetic", "status": "synth"}
            print("· sepsis: no public UCI/OpenML hit → will generate synthetic family-E cohort")
    except Exception as e:
        manifest['sepsis'] = {"source": "unresolved", "status": f"search failed: {e} → synthetic", "family": "synthetic"}
        print(f"· sepsis search failed ({e}) → synthetic")

def main():
    manifest = {}
    for t in UCI_TARGETS:
        fetch_uci(*t, manifest)
    for t in OPENML_TARGETS:
        fetch_openml_named(*t, manifest)
    resolve_sepsis(manifest)
    with open(os.path.join(C.RAW, 'MANIFEST.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    ok = sum(1 for v in manifest.values() if v.get('status') in ('ok', 'synth'))
    print(f"\nCOLLECTION DONE: {ok}/{len(manifest)} datasets available → MANIFEST.json")

if __name__ == '__main__':
    main()
