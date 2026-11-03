#!/usr/bin/env sh
set -eu

CONFIG_PATH=/data/options.json

# Read options with jq; use defaults if file missing
if [ -f "$CONFIG_PATH" ]; then
  LOG_LEVEL=$(jq -r '.log_level // "INFO"' "$CONFIG_PATH")
  BUFFER_SIZE=$(jq -r '.buffer_size // 100' "$CONFIG_PATH")
  ANALYSIS_RATE_SECONDS=$(jq -r '.analysis_rate_seconds // 300' "$CONFIG_PATH")
  ANALYSIS_MAX_LINES=$(jq -r '.analysis_max_lines // 2000' "$CONFIG_PATH")
  OPENAI_API_KEY=$(jq -r '.openai_api_key // ""' "$CONFIG_PATH")
  LLM_PROVIDER=$(jq -r '.llm_provider // "openai"' "$CONFIG_PATH")
  LLM_BASE_URL=$(jq -r '.llm_base_url // "https://router.requesty.ai/v1"' "$CONFIG_PATH")
  LLM_MODEL=$(jq -r '.llm_model // "nvidia/nemotron-3-super-120b-a12b"' "$CONFIG_PATH")
  LLM_API_KEY=$(jq -r '.llm_api_key // ""' "$CONFIG_PATH")
  LLM_HTTP_REFERER=$(jq -r '.llm_http_referer // ""' "$CONFIG_PATH")
  LLM_X_TITLE=$(jq -r '.llm_x_title // ""' "$CONFIG_PATH")
  LLM_BACKUP_PROVIDER=$(jq -r '.llm_backup_provider // "mock"' "$CONFIG_PATH")
  LLM_BACKUP_BASE_URL=$(jq -r '.llm_backup_base_url // ""' "$CONFIG_PATH")
  LLM_BACKUP_MODEL=$(jq -r '.llm_backup_model // ""' "$CONFIG_PATH")
  LLM_BACKUP_API_KEY=$(jq -r '.llm_backup_api_key // ""' "$CONFIG_PATH")
else
  LOG_LEVEL=${LOG_LEVEL:-INFO}
  BUFFER_SIZE=${BUFFER_SIZE:-100}
  ANALYSIS_RATE_SECONDS=${ANALYSIS_RATE_SECONDS:-300}
  ANALYSIS_MAX_LINES=${ANALYSIS_MAX_LINES:-2000}
  OPENAI_API_KEY=${OPENAI_API_KEY:-}
  LLM_PROVIDER=${LLM_PROVIDER:-openai}
  LLM_BASE_URL=${LLM_BASE_URL:-https://router.requesty.ai/v1}
  LLM_MODEL=${LLM_MODEL:-nvidia/nemotron-3-super-120b-a12b}
  LLM_API_KEY=${LLM_API_KEY:-}
  LLM_HTTP_REFERER=${LLM_HTTP_REFERER:-}
  LLM_X_TITLE=${LLM_X_TITLE:-}
  LLM_BACKUP_PROVIDER=${LLM_BACKUP_PROVIDER:-mock}
  LLM_BACKUP_BASE_URL=${LLM_BACKUP_BASE_URL:-}
  LLM_BACKUP_MODEL=${LLM_BACKUP_MODEL:-}
  LLM_BACKUP_API_KEY=${LLM_BACKUP_API_KEY:-}
fi

export LOG_LEVEL BUFFER_SIZE ANALYSIS_RATE_SECONDS ANALYSIS_MAX_LINES OPENAI_API_KEY
export LLM_PROVIDER LLM_BASE_URL LLM_MODEL LLM_API_KEY LLM_HTTP_REFERER LLM_X_TITLE
export LLM_BACKUP_PROVIDER LLM_BACKUP_BASE_URL LLM_BACKUP_MODEL LLM_BACKUP_API_KEY

echo "[INFO] Starting HA LLM Ops agent"
exec python3 -m agent
