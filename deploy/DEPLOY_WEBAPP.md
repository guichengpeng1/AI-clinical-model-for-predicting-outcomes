# Deploy AI Clinical Score Web App

This is the deployment path for publishing the calculator so external users can access it.

## 1. Train and export the model bundle

```bash
/home/ubuntu/miniconda3/envs/starf/bin/python /media/ubuntu/CosMx/rc_pc/20250824/scripts/train_ai_clinical_score.py \
  --input /media/ubuntu/CosMx/rc_pc/20250824/AIdata/SEER.csv \
  --output-dir /media/ubuntu/CosMx/rc_pc/20250824/outputs/ai_clinical_score
```

Expected artifact:

- `/media/ubuntu/CosMx/rc_pc/20250824/outputs/ai_clinical_score/best_model.joblib`

## 2. Prepare runtime directories

```bash
mkdir -p /media/ubuntu/CosMx/rc_pc/20250824/logs
chmod +x /media/ubuntu/CosMx/rc_pc/20250824/deploy/start_ai_clinical_score_web.sh
```

## 3. Test the app locally

```bash
HOST=127.0.0.1 PORT=8080 /media/ubuntu/CosMx/rc_pc/20250824/deploy/start_ai_clinical_score_web.sh
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Example JSON API call:

```bash
curl -X POST http://127.0.0.1:8080/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Age": 60,
    "Sex": 1,
    "Tumor Size Summary (2016+)": 5.0,
    "Grade Pathological (2018+)": 2,
    "Tsgemerge1": 2,
    "Nstagemerge1": 0,
    "Derived EOD 2018 M (2018+)": 0,
    "Race recode (W, B, AI, API)": 1,
    "Histologic Type ICD-O-3(1chRCC,2pRCC,3ccRCC)": 3
  }'
```

## 4. Run under systemd

Copy the service file:

```bash
sudo cp /media/ubuntu/CosMx/rc_pc/20250824/deploy/systemd/ai-clinical-score.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-clinical-score
sudo systemctl start ai-clinical-score
sudo systemctl status ai-clinical-score
```

## 5. Put Nginx in front

Edit the domain in:

- `/media/ubuntu/CosMx/rc_pc/20250824/deploy/nginx/ai-clinical-score.conf`

Install and enable:

```bash
sudo cp /media/ubuntu/CosMx/rc_pc/20250824/deploy/nginx/ai-clinical-score.conf /etc/nginx/sites-available/ai-clinical-score
sudo ln -s /etc/nginx/sites-available/ai-clinical-score /etc/nginx/sites-enabled/ai-clinical-score
sudo nginx -t
sudo systemctl reload nginx
```

## 6. Add HTTPS

If the server is public, obtain TLS with Certbot after DNS points to the server:

```bash
sudo certbot --nginx -d your-domain.example.com
```

## 7. What users can access

- Browser UI: `https://your-domain.example.com/`
- Health endpoint: `https://your-domain.example.com/health`
- JSON API: `https://your-domain.example.com/api/predict`

## 8. Current technical limits

- The current public app serves one locally trained model bundle.
- Predictions are only as stable as the frozen training dataset and chosen best model.
- The full manuscript algorithm list includes methods not yet available in the current local runtime: `SuperPC`, `conditional inference survival forest`, `DeepHit`, and the official `Survival Quilts` orchestration code.
