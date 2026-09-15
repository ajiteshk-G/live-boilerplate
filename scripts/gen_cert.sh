#!/usr/bin/env bash
# Generate a self-signed TLS certificate for local development.
#
# WHY YOU MIGHT NEED THIS
# -----------------------
# Chrome only grants microphone access in a "secure context". Plain HTTP on a
# remote hostname is not one, so getUserMedia() fails -- often silently.
#
# There are two ways around it. Prefer the first:
#
#   1. SSH port-forward (no certificate, no browser warning):
#        ssh -L 8080:localhost:8080 <host>
#      then open http://localhost:8080 -- localhost counts as secure.
#
#   2. This script, if you need to reach the server from another machine.
#      You will have to click through a browser warning every time, because
#      the certificate is self-signed.
#
# Usage:
#   ./scripts/gen_cert.sh [hostname]
#   uv run glive serve --host 0.0.0.0 \
#       --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key

set -euo pipefail

HOSTNAME_ARG="${1:-$(hostname -f 2>/dev/null || hostname)}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/certs"
DAYS=365

mkdir -p "$OUT_DIR"

if [[ -f "$OUT_DIR/dev.crt" ]]; then
  read -r -p "certs/dev.crt already exists. Overwrite? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Keeping the existing certificate."; exit 0; }
fi

echo "Generating a self-signed certificate for '$HOSTNAME_ARG' (valid $DAYS days)..."

# The SAN entries matter: modern browsers ignore the legacy CN field.
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$OUT_DIR/dev.key" \
  -out "$OUT_DIR/dev.crt" \
  -days "$DAYS" \
  -subj "/CN=$HOSTNAME_ARG" \
  -addext "subjectAltName=DNS:$HOSTNAME_ARG,DNS:localhost,IP:127.0.0.1" \
  2>/dev/null

chmod 600 "$OUT_DIR/dev.key"

cat <<EOF

Wrote:
  $OUT_DIR/dev.crt
  $OUT_DIR/dev.key   (mode 600)

Start the server with TLS:

  uv run glive serve --host 0.0.0.0 \\
      --ssl-certfile certs/dev.crt --ssl-keyfile certs/dev.key

Then open  https://$HOSTNAME_ARG:8080  and accept the browser warning.

certs/ is gitignored. This certificate is for local development only.
EOF
