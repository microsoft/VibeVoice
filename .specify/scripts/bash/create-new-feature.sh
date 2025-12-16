#!/usr/bin/env bash

set -e

JSON_MODE=false
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --json) JSON_MODE=true ;;
        --no-git) : ;;
        --help|-h) echo "Usage: $0 [--json] <feature_description>"; exit 0 ;;
        *) ARGS+=("$arg") ;;
    esac
done

FEATURE_DESCRIPTION="${ARGS[*]}"
if [ -z "$FEATURE_DESCRIPTION" ]; then
    echo "Usage: $0 [--json] <feature_description>" >&2
    exit 1
fi

# Function to find the repository root by searching for existing project markers
find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/.git" ] || [ -d "$dir/.specify" ]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

# Resolve repository root. Prefer git information when available, but fall back
# to searching for repository markers so the workflow still functions in repositories that
# were initialised with --no-git.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if git rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT=$(git rev-parse --show-toplevel)
    HAS_GIT=true
else
    REPO_ROOT="$(find_repo_root "$SCRIPT_DIR")"
    if [ -z "$REPO_ROOT" ]; then
        echo "Error: Could not determine repository root. Please run this script from within the repository." >&2
        exit 1
    fi
    HAS_GIT=false
fi

cd "$REPO_ROOT"

SPECS_DIR="$REPO_ROOT/specs"
mkdir -p "$SPECS_DIR"

# Detect current git branch and sanitize for directory prefix
CURRENT_BRANCH=""
BRANCH_PREFIX=""
if [ "$HAS_GIT" = true ]; then
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ -n "$CURRENT_BRANCH" ]; then
        BRANCH_PREFIX=$(echo "$CURRENT_BRANCH" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//')
    fi

    # Block protected branches (no spec creation allowed)
    lower_branch=$(echo "$CURRENT_BRANCH" | tr '[:upper:]' '[:lower:]')
    if echo "$lower_branch" | grep -Eq '^(main|dev)$|^release-v[0-9]+(\.[0-9]+)*$'; then
        echo "[specify] ERROR: Protected branch detected ('$CURRENT_BRANCH'). Spec creation is disabled on protected branches (main, dev, release-v*)." >&2
        echo "Please switch to a feature branch (issue/<ISSUE>[-summary], epic/<ISSUE>[-summary], or one ending with NNN-name)." >&2
        exit 1
    fi
fi

# Count existing feature numbers in the appropriate directory
HIGHEST=0
if [ -n "$BRANCH_PREFIX" ] && [ -d "$SPECS_DIR/$BRANCH_PREFIX" ]; then
    # Scan within branch subdirectory
    for dir in "$SPECS_DIR/$BRANCH_PREFIX"/*; do
        [ -d "$dir" ] || continue
        dirname=$(basename "$dir")
        number=$(echo "$dirname" | grep -o '^[0-9]\+' || echo "0")
        number=$((10#$number))
        if [ "$number" -gt "$HIGHEST" ]; then HIGHEST=$number; fi
    done
elif [ -d "$SPECS_DIR" ]; then
    # Fallback: scan top-level for non-git or no branch prefix
    for dir in "$SPECS_DIR"/*; do
        [ -d "$dir" ] || continue
        dirname=$(basename "$dir")
        number=$(echo "$dirname" | grep -o '^[0-9]\+' || echo "0")
        number=$((10#$number))
        if [ "$number" -gt "$HIGHEST" ]; then HIGHEST=$number; fi
    done
fi

NEXT=$((HIGHEST + 1))
FEATURE_NUM=$(printf "%03d" "$NEXT")

BRANCH_NAME=$(echo "$FEATURE_DESCRIPTION" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//')
WORDS=$(echo "$BRANCH_NAME" | tr '-' '\n' | grep -v '^$' | head -3 | tr '\n' '-' | sed 's/-$//')
BRANCH_NAME="${FEATURE_NUM}-${WORDS}"

if [ "$HAS_GIT" = true ]; then
    >&2 echo "[specify] Branch creation is disabled by policy; staying on current branch (target: $BRANCH_NAME)"
else
    >&2 echo "[specify] Git repository not detected; proceeding without branch creation (target: $BRANCH_NAME)"
fi

if [ -n "$BRANCH_PREFIX" ]; then
    FEATURE_DIR="$SPECS_DIR/$BRANCH_PREFIX/$BRANCH_NAME"
    FEATURE_KEY="$BRANCH_PREFIX/$BRANCH_NAME"
else
    FEATURE_DIR="$SPECS_DIR/$BRANCH_NAME"
    FEATURE_KEY="$BRANCH_NAME"
fi
mkdir -p "$FEATURE_DIR"

TEMPLATE="$REPO_ROOT/.specify/templates/spec-template.md"
SPEC_FILE="$FEATURE_DIR/spec.md"
if [ -f "$TEMPLATE" ]; then cp "$TEMPLATE" "$SPEC_FILE"; else touch "$SPEC_FILE"; fi

# Set the SPECIFY_FEATURE environment variable for the current session (may be nested path)
export SPECIFY_FEATURE="$FEATURE_KEY"

if $JSON_MODE; then
    printf '{"BRANCH_NAME":"%s","SPEC_FILE":"%s","FEATURE_NUM":"%s"}\n' "$BRANCH_NAME" "$SPEC_FILE" "$FEATURE_NUM"
else
    echo "BRANCH_NAME: $BRANCH_NAME"
    echo "SPEC_FILE: $SPEC_FILE"
    echo "FEATURE_NUM: $FEATURE_NUM"
    echo "SPECIFY_FEATURE environment variable set to: $BRANCH_NAME"
fi
