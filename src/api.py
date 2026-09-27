"""10 — MINIMAL HTTP API for the Triage Risk Engine (stdlib only).
EVOLVEX IT SOLUTIONS PVT. LTD.

POST /triage  {case JSON}  →  infer.py contract (risks + risk_ratio + T0-TT4)
GET  /health  →  status

Run:  python3 src/api.py 8088   (binds 0.0.0.0)
"""
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer import load_artifacts, predict_specialist, risk_block
from train_triage import rule_scan, CRITICAL_RULES
from cluster import cluster_features
import numpy as np, pandas as pd

ARTS = None

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/health'):
            self._send(200, {"ok": True, "service": "healthguard-triage-risk-engine",
                             "owner": "EVOLVEX IT SOLUTIONS PVT. LTD.",
                             "version": "triage-fusion-moe-v1"})
        else:
            self._send(404, {"error": "use POST /triage"})

    def do_POST(self):
        if not self.path.startswith('/triage'):
            return self._send(404, {"error": "use POST /triage"})
        try:
            n = int(self.headers.get('Content-Length', 0))
            case = json.loads(self.rfile.read(n) or b'{}')
        except Exception:
            return self._send(400, {"error": "invalid JSON body"})
        import infer
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # reuse the CLI pipeline
            sys.argv = ['infer.py', '-']
            import threading
        # direct call (simpler than replaying the CLI)
        row_df = pd.DataFrame([case])
        risks = {}
        for name, art in ARTS.items():
            if name == 'triage': continue
            p = predict_specialist(art, row_df)
            blk = risk_block(p, len(art['classes']))
            if blk:
                blk['model'] = 'MoE de-clustered (calibrated)'
                blk['dataset_family'] = 'real public clinical' if name != 'sepsis' else 'synthetic family-E'
                risks[name] = blk
        t = ARTS['triage']
        Xt = row_df.reindex(columns=t['features'], fill_value=0.0)
        Xaug = pd.concat([Xt.reset_index(drop=True),
                          cluster_features(Xt, t['km'], t['gmm'], t['scaler'])], axis=1)
        probs = np.zeros(len(t['classes']))
        detail = {}
        for w, (mname, m) in zip(t['weights'], t['blend']):
            pm = m.predict_proba(Xt, Xaug)[0] if type(m).__name__ == 'MoEDecluster' else m.predict_proba(Xaug)[0]
            probs += w * pm
            detail[mname] = [round(float(x), 4) for x in pm]
        pmap = {c: float(v) for c, v in zip(t['classes'], probs)}
        pred = int(max(pmap, key=pmap.get))
        s01 = pmap.get(0, 0) + pmap.get(1, 0)
        if s01 >= 0.30: pred = min(pred, 1)
        elif s01 >= 0.14: pred = min(pred, 2)
        reasons = rule_scan(row_df.iloc[0])
        if reasons:
            pred = 0 if any(r in CRITICAL_RULES for r in reasons) else min(pred, 1)
        urgent = s01 > 0.25
        review = bool((reasons and not urgent) or (urgent and not reasons and pred >= 3)) or bool(case.get('uncertain'))
        self._send(200, {
            'risk': risks,
            'triage': {
                'class': f'T{pred}',
                'meaning': ['emergency escalation', 'very urgent', 'priority consultation', 'routine consultation', 'follow-up'][pred],
                'probs': {f'T{k}': round(v, 4) for k, v in sorted(pmap.items())},
                'ensemble_votes': detail,
                'reason_codes': reasons or ['ROUTINE_STABLE'],
                'requires_human_review': review,
                'confidence': round(float(max(probs) * (1 - 0.5 * float(review))), 3),
                'rule_version': '2026-09-triage-protocol-v1',
                'model_version': 'triage-fusion-moe-v1',
            },
            'disclaimer': 'AI assistive triage — workflow urgency only, not diagnosis. Clinical decisions remain with qualified clinicians.',
        })

    def log_message(self, *a):
        pass

def main():
    global ARTS
    ARTS = load_artifacts()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print(f"HealthGuard Triage Risk Engine API on :{port} (EVOLVEX IT SOLUTIONS PVT. LTD.)")
    srv.serve_forever()

if __name__ == '__main__':
    main()
