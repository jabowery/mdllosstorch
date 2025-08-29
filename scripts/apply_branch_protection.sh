#!/usr/bin/env bash
set -euo pipefail

# Apply branch protection rules to 'main' using GitHub CLI.
# Requirements:
#   - gh CLI installed: https://cli.github.com/
#   - Admin permission on the repo
#   - GH_TOKEN or GITHUB_TOKEN env var with 'repo' scope
#
# Usage:
#   OWNER=yourname REPO=mdllosstorch ./scripts/apply_branch_protection.sh

OWNER="${OWNER:-}"
REPO="${REPO:-}"
BRANCH="${BRANCH:-main}"

if [[ -z "${OWNER}" || -z "${REPO}" ]]; then
  echo "Set OWNER and REPO env vars. Example:"
  echo "  OWNER=yourname REPO=mdllosstorch ./scripts/apply_branch_protection.sh"
  exit 1
fi

# Required status checks correspond to workflow names/jobs you want green before merging.
# Adjust contexts as needed (they must match the check names shown in PRs).
read -r -d '' PAYLOAD << 'JSON'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      { "context": "ci / test (3.10)" },
      { "context": "ci / test (3.11)" },
      { "context": "ci / test (3.12)" },
      { "context": "ci / test (3.13)" }
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": true
}
JSON

# Apply protection via REST API
gh api   -X PUT   -H "Accept: application/vnd.github+json"   "/repos/${OWNER}/${REPO}/branches/${BRANCH}/protection"   -f required_status_checks="$(echo "$PAYLOAD" | jq -c '.required_status_checks')"   -f enforce_admins=true   -f required_pull_request_reviews="$(echo "$PAYLOAD" | jq -c '.required_pull_request_reviews')"   -f restrictions=   -f allow_force_pushes=false   -f allow_deletions=false   -f required_linear_history=true

echo "Branch protection applied to ${OWNER}/${REPO}@${BRANCH}"
