#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/ubuntu/miniconda3/envs/starf/bin/python"
MODEL_BUNDLE="${ROOT_DIR}/outputs/ai_clinical_score/best_model.joblib"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python runtime not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_BUNDLE}" ]]; then
  echo "Model bundle not found: ${MODEL_BUNDLE}" >&2
  echo "Run training first:" >&2
  echo "  ${PYTHON_BIN} ${ROOT_DIR}/scripts/train_ai_clinical_score.py --input ${ROOT_DIR}/AIdata/SEER.csv --output-dir ${ROOT_DIR}/outputs/ai_clinical_score" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/webapp/ai_clinical_score_server.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-bundle "${MODEL_BUNDLE}"
