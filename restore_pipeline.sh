#!/usr/bin/env bash
# Regenerate every Q.6 artifact from code. The pipeline is fully seeded, so this
# reproduces the exact numbers recorded in PIPELINE_REPORT.md.
# Runtime ~5 min on 2 vCPU.
set -e
cd /home/user/healthguard-ml
# installed packages do not survive workspace snapshots - make the restore self-sufficient
python3 - <<'EOF' || true
import importlib
missing=[m for m in ['pyarrow','xgboost','lightgbm'] if not importlib.util.find_spec(m)]
if missing:
    import subprocess, os
    env=dict(os.environ); env['TMPDIR']='/var/tmp'
    subprocess.run(['pip','install','--quiet','--no-cache-dir']+missing, env=env, check=True)
    print('installed:', missing)
EOF
mkdir -p data/pipeline reports/pipeline models/pipeline

echo "=== 01 collection ==="
python3 -u pipeline/01_collect.py 2>&1 | tail -4

echo "=== 02 embedding (numeric) ==="
python3 -u pipeline/02_embed.py --method numeric 2>&1 | tail -3

echo "=== 03 dedup + clustering (k=4000, cap=50) ==="
python3 -u pipeline/03_dedup_cluster.py --method numeric --algo kmeans --clusters 4000 --cap 50 2>&1 | tail -5

echo "=== 04 cluster-disjoint splits ==="
python3 -u pipeline/04_sample.py --method numeric --algo kmeans --clusters 4000 2>&1 | tail -6

echo "=== 05 fine-tune: curated vs raw control ==="
python3 -u pipeline/05_finetune.py --model lgbm --dataset train_curated     --tag lgbm_curated 2>&1 | grep -E '"macro_f1"|"T0_any_down"|external_real_rule_agreement'
python3 -u pipeline/05_finetune.py --model lgbm --dataset train_raw_control --tag lgbm_control 2>&1 | grep -E '"macro_f1"|"T0_any_down"|external_real_rule_agreement'

echo "=== 06 error clustering + targeted re-collection ==="
python3 -u pipeline/06_refine.py 2>&1 | grep -E '"before"|"after"|macro_f1|macro_f1_inflation' | head -12

echo "=== 07 efficiency (distillation) ==="
python3 -u pipeline/07_efficiency.py 2>&1 | grep -E 'size_mb|throughput|macro_f1|total_usd' | head -12

echo "=== trimming regenerable intermediates (keep the workspace inside the snapshot cap) ==="
rm -f data/pipeline/emb_numeric.npy data/pipeline/clustered_*.parquet data/pipeline/raw_synth_site_*.parquet
du -sh data/pipeline reports/pipeline models/pipeline
echo "RESTORE COMPLETE"
