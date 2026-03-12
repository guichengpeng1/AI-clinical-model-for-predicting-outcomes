#!/home/ubuntu/miniconda3/envs/starf/bin/python
import argparse
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd


FIELD_SPECS = [
    ("Age", "Age (18-90)", "number", None, "18", "90", "1"),
    ("Sex", "Sex", "select", [("0", "Female"), ("1", "Male")], None, None, None),
    (
        "Tumor Size Summary (2016+)",
        "Tumor Size (cm)",
        "number",
        None,
        "0",
        "300",
        "0.1",
    ),
    (
        "Grade Pathological (2018+)",
        "Pathologic Grade",
        "select",
        [("1", "Grade 1"), ("2", "Grade 2"), ("3", "Grade 3"), ("4", "Grade 4")],
        None,
        None,
        None,
    ),
    ("Tsgemerge1", "T Stage", "select", [("1", "T1"), ("2", "T2"), ("3", "T3"), ("4", "T4"), ("5", "Other/Tx")], None, None, None),
    ("Nstagemerge1", "N Stage", "select", [("0", "N0"), ("1", "N1"), ("2", "Nx/Unknown")], None, None, None),
    (
        "Derived EOD 2018 M (2018+)",
        "M Stage",
        "select",
        [("0", "M0"), ("1", "M1"), ("2", "Mx/Unknown")],
        None,
        None,
        None,
    ),
    (
        "Race recode (W, B, AI, API)",
        "Race",
        "select",
        [
            ("1", "White"),
            ("2", "Black"),
            ("3", "American Indian / Alaska Native"),
            ("4", "Asian / Pacific Islander"),
            ("5", "Other"),
        ],
        None,
        None,
        None,
    ),
    (
        "Histologic Type ICD-O-3(1chRCC,2pRCC,3ccRCC)",
        "Histologic Subtype",
        "select",
        [("1", "Chromophobe RCC"), ("2", "Papillary RCC"), ("3", "Clear cell RCC")],
        None,
        None,
        None,
    ),
]


def load_bundle(path):
    bundle = joblib.load(path)
    return bundle


def render_form(values, prediction, model_status):
    fields_html = []
    for key, label, input_type, options, min_v, max_v, step in FIELD_SPECS:
        value = values.get(key, "")
        if input_type == "select":
            option_html = []
            for opt_value, opt_label in options:
                selected = " selected" if str(value) == opt_value else ""
                option_html.append(f'<option value="{html.escape(opt_value)}"{selected}>{html.escape(opt_label)}</option>')
            control = f"""
                <select id="{html.escape(key)}" name="{html.escape(key)}">
                    {''.join(option_html)}
                </select>
            """
        else:
            control = (
                f'<input id="{html.escape(key)}" name="{html.escape(key)}" type="number" '
                f'value="{html.escape(str(value))}" min="{min_v}" max="{max_v}" step="{step}" required>'
            )
        fields_html.append(
            f"""
            <label class="field">
                <span>{html.escape(label)}</span>
                {control}
            </label>
            """
        )

    result_block = ""
    if prediction:
        risk_cards = []
        for label, value in prediction["horizon_risks"].items():
            risk_cards.append(
                f"""
                <div class="metric">
                    <div class="metric-label">{html.escape(label)}</div>
                    <div class="metric-value">{value:.1%}</div>
                </div>
                """
            )
        result_block = f"""
        <section class="results">
            <h2>Prediction</h2>
            <div class="hero-metric">
                <div class="metric-label">Raw risk score</div>
                <div class="hero-value">{prediction['raw_risk']:.4f}</div>
            </div>
            <div class="metrics-grid">
                {''.join(risk_cards)}
            </div>
            <p class="note">
                Model: {html.escape(prediction['model_name'])}. Risk horizons are derived from the trained SEER-based pipeline.
            </p>
        </section>
        """

    page = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>AI Clinical Score Calculator</title>
      <style>
        :root {{
          --bg: #f3efe6;
          --ink: #1d2a26;
          --panel: rgba(255,255,255,0.78);
          --edge: rgba(29,42,38,0.14);
          --accent: #a83f2f;
          --accent-2: #2d6a57;
          --shadow: 0 24px 80px rgba(34, 44, 38, 0.14);
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: Georgia, "Times New Roman", serif;
          color: var(--ink);
          background:
            radial-gradient(circle at top left, rgba(168,63,47,0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(45,106,87,0.18), transparent 35%),
            linear-gradient(135deg, #f8f4ea 0%, #ece4d6 100%);
          min-height: 100vh;
        }}
        .shell {{
          width: min(1120px, calc(100% - 32px));
          margin: 32px auto;
          display: grid;
          grid-template-columns: 1.2fr 0.8fr;
          gap: 20px;
        }}
        .card {{
          background: var(--panel);
          backdrop-filter: blur(8px);
          border: 1px solid var(--edge);
          border-radius: 24px;
          box-shadow: var(--shadow);
          padding: 28px;
        }}
        h1, h2 {{ margin: 0 0 12px; line-height: 1; }}
        h1 {{ font-size: clamp(2rem, 5vw, 4.2rem); letter-spacing: -0.04em; }}
        h2 {{ font-size: 1.3rem; }}
        p {{ margin: 0 0 12px; line-height: 1.5; }}
        .lede {{ color: rgba(29,42,38,0.74); max-width: 58ch; }}
        .status {{
          display: inline-block;
          margin-top: 16px;
          padding: 10px 14px;
          border-radius: 999px;
          background: rgba(45,106,87,0.1);
          border: 1px solid rgba(45,106,87,0.2);
          font-size: 0.95rem;
        }}
        form {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px 16px;
          margin-top: 24px;
        }}
        .field {{
          display: grid;
          gap: 8px;
          font-size: 0.95rem;
        }}
        .field span {{
          font-weight: 600;
        }}
        input, select {{
          width: 100%;
          border: 1px solid var(--edge);
          border-radius: 14px;
          padding: 12px 14px;
          font: inherit;
          color: var(--ink);
          background: rgba(255,255,255,0.92);
        }}
        .actions {{
          grid-column: 1 / -1;
          display: flex;
          gap: 12px;
          align-items: center;
          margin-top: 6px;
        }}
        button {{
          border: 0;
          border-radius: 999px;
          padding: 14px 20px;
          font: inherit;
          font-weight: 700;
          background: linear-gradient(135deg, var(--accent), #cb604a);
          color: white;
          cursor: pointer;
        }}
        .secondary {{
          color: rgba(29,42,38,0.72);
          font-size: 0.9rem;
        }}
        .results {{
          display: grid;
          gap: 16px;
        }}
        .hero-metric {{
          padding: 18px 20px;
          border-radius: 20px;
          background: linear-gradient(135deg, rgba(168,63,47,0.12), rgba(45,106,87,0.12));
        }}
        .metric-label {{
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: rgba(29,42,38,0.66);
        }}
        .hero-value {{
          font-size: 2.8rem;
          line-height: 1;
          margin-top: 10px;
        }}
        .metrics-grid {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }}
        .metric {{
          border: 1px solid var(--edge);
          border-radius: 18px;
          padding: 16px;
          background: rgba(255,255,255,0.66);
        }}
        .metric-value {{
          font-size: 1.8rem;
          margin-top: 8px;
        }}
        .note {{
          color: rgba(29,42,38,0.72);
          font-size: 0.95rem;
        }}
        @media (max-width: 880px) {{
          .shell {{ grid-template-columns: 1fr; }}
          form {{ grid-template-columns: 1fr; }}
        }}
      </style>
    </head>
    <body>
      <main class="shell">
        <section class="card">
          <h1>AI Clinical Score</h1>
          <p class="lede">
            Enter the nine clinicopathologic variables used by the RCC SEER training pipeline to estimate risk.
            This page is wired to the locally trained model bundle.
          </p>
          <div class="status">{html.escape(model_status)}</div>
          <form method="post" action="/predict">
            {''.join(fields_html)}
            <div class="actions">
              <button type="submit">Calculate Score</button>
              <div class="secondary">Use the same variable encoding as the training cohort.</div>
            </div>
          </form>
        </section>
        <aside class="card">
          {result_block if result_block else '<section class="results"><h2>Prediction</h2><p class="note">Submit the form to calculate the score and horizon-specific risks.</p></section>'}
        </aside>
      </main>
    </body>
    </html>
    """
    return page.encode("utf-8")


class AppHandler(BaseHTTPRequestHandler):
    bundle = None
    bundle_path = None

    def _model_status(self):
        if self.bundle is None:
            return f"Model bundle not found at {self.bundle_path}. Train the model first."
        return f"Loaded model bundle: {self.bundle['best_model_name']}"

    def _send_html(self, content, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _coerce_payload(self, payload):
        row = {}
        for key, _, _, *_ in FIELD_SPECS:
            raw = str(payload.get(key, "")).strip()
            if raw == "":
                raise ValueError(f"missing value for {key}")
            row[key] = float(raw) if key in ["Age", "Tumor Size Summary (2016+)"] else str(int(float(raw)))
        return row

    def _predict(self, row):
        if self.bundle is None:
            raise RuntimeError("model bundle is not loaded")
        df = pd.DataFrame([row])
        model = self.bundle["model"]
        raw_risk = float(model.predict(df)[0])
        horizon_risks = {}
        if hasattr(model, "predict_survival_function"):
            fn = model.predict_survival_function(df)[0]
            for horizon in self.bundle["horizons"]:
                horizon_risks[f"{int(horizon)}-month risk"] = 1.0 - float(fn(horizon))
        return {
            "raw_risk": raw_risk,
            "horizon_risks": horizon_risks,
            "model_name": self.bundle["best_model_name"],
        }

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            payload = {
                "ok": self.bundle is not None,
                "model_loaded": self.bundle is not None,
                "bundle_path": str(self.bundle_path),
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_html(render_form({}, None, self._model_status()))

    def do_POST(self):
        if self.path == "/api/predict":
            if self.bundle is None:
                self._send_json(
                    {"ok": False, "error": self._model_status()},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                row = self._coerce_payload(payload)
                prediction = self._predict(row)
                self._send_json({"ok": True, "input": row, "prediction": prediction})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if self.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = {k: v[0] for k, v in parse_qs(body).items()}

        if self.bundle is None:
            self._send_html(render_form(form, None, self._model_status()), status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        try:
            row = self._coerce_payload(form)
            prediction = self._predict(row)
            self._send_html(render_form(form, prediction, self._model_status()))
        except Exception as exc:
            status = f"{self._model_status()} Input error: {exc}"
            self._send_html(render_form(form, None, status), status=HTTPStatus.BAD_REQUEST)


def main():
    parser = argparse.ArgumentParser(description="Serve the AI clinical score calculator.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-bundle", default="outputs/ai_clinical_score/best_model.joblib")
    args = parser.parse_args()

    AppHandler.bundle_path = Path(args.model_bundle)
    if AppHandler.bundle_path.exists():
        AppHandler.bundle = load_bundle(AppHandler.bundle_path)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Serving AI clinical score calculator on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
