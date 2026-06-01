#!/usr/bin/env bash
# scripts/cleanup_merged_branches.sh
#
# Delete fully-merged remote branches from the EasyOPD origin.
#
# Usage:
#   bash scripts/cleanup_merged_branches.sh                # dry-run
#   bash scripts/cleanup_merged_branches.sh --execute      # actually delete
#
# Requirements:
#   * `git` available locally with `origin` pointing to the EasyOPD GitHub repo
#   * `gh` CLI authenticated (`gh auth login`) for the API call;
#     OR set GITHUB_TOKEN and GH_REPO env vars to fall back to `curl`.
#
# Safety:
#   * For each candidate branch we re-verify `ahead==0` against `origin/main`.
#   * If `ahead != 0` (i.e. there is unmerged work), the branch is REPORTED
#     and NEVER deleted; the maintainer must review manually.
#   * Official version branches (`release/*`, `v[0-9]*.x`, `revert-*`, `main`,
#     `HEAD`) are NOT touched.

set -euo pipefail

CANDIDATES=(
  "GOPD"
  "qy/GKD"
  "qy/OPCD"
  "qy/SDPO"
  "qy/update-readme-links"
  "SOD_V1"
  "sunjie-simct"
  "visionOPD"
)

# --- arg parsing -----------------------------------------------------------
EXECUTE=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    -h|--help)
      grep -E "^# " "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# --- 1. fetch & sync --------------------------------------------------------
echo "==> git fetch origin --prune"
git fetch origin --prune

# --- 2. determine GitHub owner/repo ----------------------------------------
if [[ -n "${GH_REPO:-}" ]]; then
  REPO="$GH_REPO"
else
  ORIGIN_URL="$(git remote get-url origin)"
  # Match either git@github.com:owner/repo(.git) or https://github.com/owner/repo(.git)
  REPO="$(printf '%s' "$ORIGIN_URL" \
    | sed -E 's#^.*github\.com[:/]([^/]+/[^/.]+)(\.git)?$#\1#')"
fi
echo "==> Target GitHub repo: ${REPO}"

# --- 3. validate each candidate --------------------------------------------
SAFE_TO_DELETE=()
UNSAFE=()
MISSING=()

for br in "${CANDIDATES[@]}"; do
  if ! git show-ref --verify --quiet "refs/remotes/origin/${br}"; then
    MISSING+=("${br}")
    continue
  fi
  ahead=$(git rev-list --count "origin/main..origin/${br}")
  behind=$(git rev-list --count "origin/${br}..origin/main")
  if [[ "$ahead" == "0" ]]; then
    SAFE_TO_DELETE+=("${br}")
    printf '  [OK ] %-30s ahead=%s behind=%s — fully merged, will delete\n' "$br" "$ahead" "$behind"
  else
    UNSAFE+=("${br}")
    printf '  [WARN] %-30s ahead=%s behind=%s — UNMERGED, refusing to delete\n' "$br" "$ahead" "$behind"
  fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo
  echo "==> Branches no longer on remote (already cleaned up):"
  for br in "${MISSING[@]}"; do echo "    - ${br}"; done
fi

if [[ ${#UNSAFE[@]} -gt 0 ]]; then
  echo
  echo "==> The following branches have UNMERGED commits and were SKIPPED:"
  for br in "${UNSAFE[@]}"; do echo "    - ${br}"; done
  echo "    Maintainer must review them manually before re-running."
fi

if [[ ${#SAFE_TO_DELETE[@]} -eq 0 ]]; then
  echo
  echo "==> Nothing safe to delete. Exiting."
  exit 0
fi

# --- 4. dry-run vs execute -------------------------------------------------
if [[ "$EXECUTE" -ne 1 ]]; then
  echo
  echo "==> DRY RUN. Re-run with --execute to actually delete the ${#SAFE_TO_DELETE[@]} branch(es)."
  exit 0
fi

# --- 5. delete via gh CLI (preferred) or curl fallback ---------------------
delete_branch() {
  local br="$1"
  if command -v gh >/dev/null 2>&1; then
    # gh API auto-handles slashes in branch names
    gh api -X DELETE "repos/${REPO}/git/refs/heads/${br}" \
      && echo "    deleted via gh: ${br}" \
      || { echo "    FAILED via gh: ${br}" >&2; return 1; }
  elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
    local enc_br
    enc_br="$(printf '%s' "$br" | sed 's#/#%2F#g')"
    curl -fsS -X DELETE \
      -H "Authorization: token ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      "https://api.github.com/repos/${REPO}/git/refs/heads/${enc_br}" \
      && echo "    deleted via curl: ${br}" \
      || { echo "    FAILED via curl: ${br}" >&2; return 1; }
  else
    echo "    ERROR: neither 'gh' CLI nor GITHUB_TOKEN env var is available." >&2
    return 1
  fi
}

echo
echo "==> Deleting ${#SAFE_TO_DELETE[@]} branch(es)…"
for br in "${SAFE_TO_DELETE[@]}"; do
  delete_branch "$br"
done

echo
echo "==> Re-syncing local refs"
git fetch origin --prune

echo
echo "==> Done."
