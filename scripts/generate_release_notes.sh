#!/usr/bin/env bash
set -euo pipefail

# Usage:
#  BASE_URL=https://gpt.lazarev.cloud/ollama/v1 \
#  API_KEY=sk-xxxx \
#  MODEL=qwen3-coder:latest \
#  ./scripts/generate_release_notes.sh > RELEASE_NOTES.md

BASE_URL=${BASE_URL:-"https://gpt.lazarev.cloud/ollama/v1"}
MODEL=${MODEL:-"qwen3-coder:latest"}
MAX_COMMITS=${MAX_COMMITS:-200}

if [[ -z "${API_KEY:-}" ]]; then
  echo "ERROR: API_KEY env var is required" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Installing jq..." >&2
  sudo apt-get update -y >/dev/null 2>&1 && sudo apt-get install -y jq >/dev/null 2>&1
fi

last_tag=$(git describe --tags --abbrev=0 2>/dev/null || true)
if [[ -n "$last_tag" ]]; then
  range="$last_tag..HEAD"
else
  range=""
fi

if [[ -n "$range" ]]; then
  commits=$(git log "$range" --pretty=format:'- %s' -n "$MAX_COMMITS")
else
  commits=$(git log --pretty=format:'- %s' -n "$MAX_COMMITS")
fi

system_msg="You are an expert technical writer who creates accurate, succinct release notes."
read -r -d '' user_prompt <<'EOF'
Draft concise, high-quality release notes for the next version of the LCM Image Generator.

Requirements:
- Use markdown with sections: Added, Changed, Fixed, Security, CI/CD, Docker, Docs.
- Mention GHCR images: `ghcr.io/lazarev-cloud/lcm:latest` and `ghcr.io/lazarev-cloud/lcm:gpu`.
- Include "Quick start" code blocks for CPU and GPU run commands.
- Note FastAPI web UI, `/healthz`, non-root user, OCI labels, and Docker hardening where relevant.
- Keep to ~200–300 words, no placeholders.
EOF

payload=$(jq -n \
  --arg model "$MODEL" \
  --arg system "$system_msg" \
  --arg user   "$user_prompt

Commit summary:
$commits" \
  '{model:$model, temperature:0.2, stream:false, messages:[{role:"system", content:$system}, {role:"user", content:$user}]}'
)

retries=3
for attempt in $(seq 1 $retries); do
  resp=$(curl -sS -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
    -d "$payload" "$BASE_URL/chat/completions" || true)

  content=$(printf '%s' "$resp" | jq -r '.choices[0].message.content // empty' 2>/dev/null || true)
  if [[ -n "$content" ]]; then
    printf '%s\n' "$content"
    exit 0
  fi
  sleep 2
done

echo "ERROR: Failed to obtain release notes from model" >&2
echo "$resp" >&2
exit 1


