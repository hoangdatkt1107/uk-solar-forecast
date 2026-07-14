#!/usr/bin/env bash
# Trigger one serve run immediately (don't wait for the hourly cron) — e.g. to seed
# the very first forecast right after deploy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

echo "==> Starting job ${PREFIX}-serve"
az containerapp job start -g "$RG" -n "${PREFIX}-serve" -o table
echo ""
echo "Follow logs:"
echo "  az containerapp job execution list -g $RG -n ${PREFIX}-serve -o table"
