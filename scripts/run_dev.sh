#!/usr/bin/env bash
# Start the dev server after checking the things that usually go wrong.
#
# Usage:  ./scripts/run_dev.sh [-- extra glive serve args]

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

UV="${UV:-uv}"
command -v "$UV" >/dev/null 2>&1 || UV="$HOME/.local/bin/uv"
command -v "$UV" >/dev/null 2>&1 || {
  echo "error: uv not found. Install it or set UV=/path/to/uv." >&2
  exit 1
}

# --- project -----------------------------------------------------------------
if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]] && [[ ! -f .env ]]; then
  cat >&2 <<'EOF'
error: GOOGLE_CLOUD_PROJECT is not set and there is no .env file.

  cp .env.example .env     # then edit it
  export GOOGLE_CLOUD_PROJECT=your-project-id
EOF
  exit 1
fi

# --- credentials -------------------------------------------------------------
# Vertex uses Application Default Credentials, not an API key. A stale ADC file
# produces a confusing 401 much later, so warn about it now.
ADC="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
if [[ ! -f "$ADC" ]]; then
  echo "warning: no Application Default Credentials found at $ADC" >&2
  echo "         run: gcloud auth application-default login" >&2
elif [[ -n "$(find "$ADC" -mtime +7 2>/dev/null)" ]]; then
  echo "warning: ADC file is over a week old; refresh it if you hit a 401:" >&2
  echo "         gcloud auth application-default login" >&2
fi

# --- sync and validate -------------------------------------------------------
echo "==> Syncing dependencies"
"$UV" sync --extra dev --quiet

echo "==> Validating configuration"
"$UV" run glive validate-config >/dev/null

PORT="$("$UV" run python -c "
from gemini_live.settings.loader import load_config
print(load_config('config/config.yaml').server.port)
" 2>/dev/null || echo 8080)"

cat <<EOF

==> Starting server on port $PORT

    Open  http://localhost:$PORT

    Chrome blocks the microphone on a non-secure origin. If this machine is
    remote, forward the port from your laptop first:

        ssh -L $PORT:localhost:$PORT $(hostname -f 2>/dev/null || hostname)

    and use localhost -- not the remote hostname.

EOF

exec "$UV" run glive serve "$@"
