# AI Clinical Score Web App

This directory contains a lightweight HTTP server for the AI-clinical score calculator.

## Run

Train the model first:

```bash
/home/ubuntu/miniconda3/envs/starf/bin/python scripts/train_ai_clinical_score.py \
  --input AIdata/SEER.csv \
  --output-dir outputs/ai_clinical_score
```

Then start the web server:

```bash
/home/ubuntu/miniconda3/envs/starf/bin/python webapp/ai_clinical_score_server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --model-bundle outputs/ai_clinical_score/best_model.joblib
```

Open:

```text
http://<server-ip>:8080
```

## Notes

- The page currently loads the locally trained SEER-based model bundle.
- The form uses the same variable schema as the training pipeline.
- If you want global public access, the next deployment step is to place this service behind a public reverse proxy or cloud VM.
