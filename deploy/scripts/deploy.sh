#!/usr/bin/env bash
# start by:
#   export GRIDSIGHT_HF_TOKEN=hf_xxx
#   cp deploy/scripts/config.env.example deploy/scripts/config.env  # edit it
#   ./deploy/scripts/deploy.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

: "${GRIDSIGHT_HF_TOKEN:?set GRIDSIGHT_HF_TOKEN in your shell (export GRIDSIGHT_HF_TOKEN=hf_...)}"
: "${GRIDSIGHT_BRONZE_HF_REPO:?set GRIDSIGHT_BRONZE_HF_REPO in config.env}"

echo "==> Registering resource providers (first-time subscriptions need this)"
for ns in Microsoft.ContainerRegistry Microsoft.App Microsoft.OperationalInsights \
          Microsoft.KeyVault Microsoft.Storage Microsoft.ManagedIdentity; do
  state="$(az provider show -n "$ns" --query registrationState -o tsv 2>/dev/null || echo NotRegistered)"
  if [ "$state" != "Registered" ]; then
    echo "    registering $ns ..."
    az provider register --namespace "$ns" --wait -o none
  fi
done

echo "==> Resource group: $RG"
# so reuse an existing RG regardless of the region it was first created in
if az group show -n "$RG" -o none 2>/dev/null; then
  echo "    exists in $(az group show -n "$RG" --query location -o tsv), reusing"
else
  az group create -n "$RG" -l "$LOCATION" -o none
fi

echo "==> Container registry: $ACR"
if az acr show -n "$ACR" -g "$RG" -o none 2>/dev/null; then
  echo "    exists, reusing"
else
  # errors surface here 
  az acr create -n "$ACR" -g "$RG" --sku Basic --admin-enabled false -l "$LOCATION" -o none
fi
ACR_LOGIN="$(az acr show -n "$ACR" -g "$RG" --query loginServer -o tsv)"

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  echo "==> Skipping image build (SKIP_BUILD=1) — using images already in ACR"
else
  echo "==> Building images locally (linux/amd64) and pushing to ACR"
  # ACR Tasks (az acr build) is blocked on restricted subscriptions, so build here instead.
  # Container Apps runs amd64, so force the platform even on Apple Silicon.
  command -v docker >/dev/null 2>&1 || {
    echo "ERROR: Docker not found. Cloud build (ACR Tasks) is blocked on this subscription,"
    echo "so images must be built locally. Start Docker Desktop and re-run — or ask me to"
    echo "switch to the GitHub Actions -> GHCR build (no local Docker, native amd64)."
    exit 1
  }
  az acr login -n "$ACR"
  docker buildx build --platform linux/amd64 -f deploy/Dockerfile.api \
    -t "$ACR_LOGIN/gridsight-api:$TAG" --push .
  docker buildx build --platform linux/amd64 -f deploy/Dockerfile.job \
    -t "$ACR_LOGIN/gridsight-job:$TAG" --push .
fi

echo "==> Deploying infrastructure (Bicep)"
az deployment group create \
  -g "$RG" \
  -f deploy/bicep/main.bicep \
  -p namePrefix="$PREFIX" \
     location="$LOCATION" \
     acrName="$ACR" \
     apiImage="$ACR_LOGIN/gridsight-api:$TAG" \
     jobImage="$ACR_LOGIN/gridsight-job:$TAG" \
     bronzeHfRepo="$GRIDSIGHT_BRONZE_HF_REPO" \
     hfToken="$GRIDSIGHT_HF_TOKEN" \
     serveCron="$SERVE_CRON" \
     retrainCron="$RETRAIN_CRON" \
     enableRetrain="${ENABLE_RETRAIN:-true}" \
  -o none

API_URL="$(az deployment group show -g "$RG" -n main --query properties.outputs.apiUrl.value -o tsv 2>/dev/null || true)"
echo ""
echo "==> Done."
echo "    API:      ${API_URL:-<check: az containerapp show -g $RG -n ${PREFIX}-api>}"
echo "    Health:   ${API_URL:+$API_URL/health}"
echo ""
echo "Next:"
echo "  * Seed the first forecast now:   ./deploy/scripts/run_serve_now.sh"
echo "  * Deploy the dashboard to SWA:   ./deploy/scripts/deploy_frontend.sh ${API_URL:-<api-url>}"
echo "    (passing the API URL overlays the live next-12h forecast on the dashboard)"
