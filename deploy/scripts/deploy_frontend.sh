#!/usr/bin/env bash
# Deploy the static dashboard (index.html + history.json) to Azure Static Web Apps.
#
#   ./deploy/scripts/deploy_frontend.sh [API_URL]
#
# The dashboard loads the 2-year history.json backtest. Pass the forecast API URL to
# also overlay the real next-12h forecast at the right edge (it fetches /forecast live).
# Needs Node (npx) for the SWA CLI. Regenerate history.json first if the model changed:
#   env/bin/python frontend/build_history.py
set -euo pipefail

API_URL="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

[ -f frontend/history.json ] || { echo "frontend/history.json missing — run: env/bin/python frontend/build_history.py"; exit 1; }

SWA_NAME="${PREFIX}-web"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "==> Staging frontend (index.html, history.json, staticwebapp.config.json)"
cp frontend/index.html frontend/history.json frontend/staticwebapp.config.json "$BUILD_DIR/"
if [ -n "$API_URL" ]; then
  echo "==> Wiring live forecast overlay -> $API_URL"
  sed -i.bak "s#window.GRIDSIGHT_API_BASE = window.GRIDSIGHT_API_BASE || \"\";#window.GRIDSIGHT_API_BASE = \"$API_URL\";#" \
    "$BUILD_DIR/index.html" && rm -f "$BUILD_DIR/index.html.bak"
fi

# Static Web Apps only exists in a few regions (centralus, eastus2, westus2, westeurope,
# eastasia) — NOT spaincentral, where the rest of the stack lives. It's a global CDN, so the
# region only decides where the app's metadata sits; westeurope is closest to the UK.
SWA_LOCATION="${SWA_LOCATION:-westeurope}"

# Static Web Apps needs the Microsoft.Web provider; deploy.sh doesn't register it (the API
# and jobs use Microsoft.App instead), so a fresh subscription fails here without this.
if [ "$(az provider show -n Microsoft.Web --query registrationState -o tsv 2>/dev/null)" != "Registered" ]; then
  echo "==> Registering resource provider Microsoft.Web (first time only)"
  az provider register --namespace Microsoft.Web --wait -o none
fi

echo "==> Creating Static Web App (Free) if needed: $SWA_NAME ($SWA_LOCATION)"
if az staticwebapp show -n "$SWA_NAME" -g "$RG" -o none 2>/dev/null; then
  echo "    exists, reusing"
else
  # don't swallow errors here — a failed create used to surface later as a confusing
  # "ResourceNotFound" on the secrets call
  az staticwebapp create -n "$SWA_NAME" -g "$RG" -l "$SWA_LOCATION" --sku Free -o none
fi
TOKEN="$(az staticwebapp secrets list -n "$SWA_NAME" -g "$RG" --query properties.apiKey -o tsv)"

echo "==> Deploying content"
npx -y @azure/static-web-apps-cli deploy "$BUILD_DIR" \
  --deployment-token "$TOKEN" --env production

URL="$(az staticwebapp show -n "$SWA_NAME" -g "$RG" --query defaultHostname -o tsv)"
echo ""
echo "==> Frontend live: https://$URL"
echo "    (Remember to add https://$URL to the API CORS if you locked it down.)"
