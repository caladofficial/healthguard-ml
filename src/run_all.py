"""00 — ORCHESTRATOR: collect → harmonize → cluster → specialists → triage.
EVOLVEX IT SOLUTIONS PVT. LTD.
"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ('collect', 'collect.py'),
    ('harmonize', 'datasets.py'),
    ('clustering', 'cluster.py'),
    ('specialists', 'train_specialists.py'),
    ('triage', 'train_triage.py'),
]

def main():
    for name, script in STEPS:
        print(f"\n{'='*70}\nSTEP: {name}\n{'='*70}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, script)])
        if r.returncode != 0:
            print(f"STEP {name} FAILED (exit {r.returncode}) — continuing where safe")
    print("\nPIPELINE COMPLETE. Try:  python3 src/infer.py case_example.json", flush=True)

if __name__ == '__main__':
    main()
