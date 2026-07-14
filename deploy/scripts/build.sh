#!/usr/bin/env bash
# Rebuild the images in ACR and roll them out (after a code change)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/config.env"

ACR_LOGIN="$(az acr show -n "$ACR" -g "$RG" --query loginServer -o tsv)"

echo "==> Rebuilding images"
az acr build -r "$ACR" -f deploy/Dockerfile.api -t "gridsight-api:$TAG" . -o none
az acr build -r "$ACR" -f deploy/Dockerfile.job -t "gridsight-job:$TAG" . -o none

echo "==> Rolling out"
az containerapp update -g "$RG" -n "${PREFIX}-api" \
  --image "$ACR_LOGIN/gridsight-api:$TAG" -o none
az containerapp job update -g "$RG" -n "${PREFIX}-serve" \
  --image "$ACR_LOGIN/gridsight-job:$TAG" -o none
az containerapp job update -g "$RG" -n "${PREFIX}-retrain" \
  --image "$ACR_LOGIN/gridsight-job:$TAG" -o none 2>/dev/null || true

echo "==> Done."
