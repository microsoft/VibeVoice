#!/usr/bin/env bash
# Common functions and variables for all scripts

# Get repository root, with fallback for non-git repositories
get_repo_root() {
    if git rev-parse --show-toplevel >/dev/null 2>&1; then
        git rev-parse --show-toplevel
    else
        # Fall back to script location for non-git repos
        local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        (cd "$script_dir/../../.." && pwd)
    fi
}

# Get current branch, with fallback for non-git repositories
get_current_branch() {
    # First check if SPECIFY_FEATURE environment variable is set
    if [[ -n "${SPECIFY_FEATURE:-}" ]]; then
        echo "$SPECIFY_FEATURE"
        return
    fi

    # Then check git if available
    if git rev-parse --abbrev-ref HEAD >/dev/null 2>&1; then
        git rev-parse --abbrev-ref HEAD
        return
    fi

    # For non-git repos, try to find the latest feature directory
    local repo_root=$(get_repo_root)
    local specs_dir="$repo_root/specs"

    if [[ -d "$specs_dir" ]]; then
        local latest_feature=""
        local highest=0

        for dir in "$specs_dir"/*; do
            if [[ -d "$dir" ]]; then
                local dirname=$(basename "$dir")
                if [[ "$dirname" =~ ^([0-9]{3})- ]]; then
                    local number=${BASH_REMATCH[1]}
                    number=$((10#$number))
                    if [[ "$number" -gt "$highest" ]]; then
                        highest=$number
                        latest_feature=$dirname
                    fi
                fi
            fi
        done

        if [[ -n "$latest_feature" ]]; then
            echo "$latest_feature"
            return
        fi
    fi

    echo "main"  # Final fallback
}

# Check if we have git available
has_git() {
    git rev-parse --show-toplevel >/dev/null 2>&1
}

check_feature_branch() {
    local branch="$1"
    local has_git_repo="$2"

    # For non-git repos, we can't enforce branch naming but still provide output
    if [[ "$has_git_repo" != "true" ]]; then
        echo "[specify] Warning: Git repository not detected; skipped branch validation" >&2
        return 0
    fi

    # Allow nested feature keys like "issue-767/001-feature-name"
    local branch_last_segment="$branch"
    if [[ "$branch" == *"/"* ]]; then
        branch_last_segment="${branch##*/}"
    fi

    # Accept patterns based on repository rules:
    # - issue/<ISSUE>[-summary]
    # - epic/<ISSUE>[-summary]
    # - Or classic feature key ending with NNN-*
    local branch_lower
    branch_lower=$(echo "$branch" | tr '[:upper:]' '[:lower:]')

    local valid="false"
    local protected="false"

    # Classic feature key (last segment starts with 3-digit prefix)
    if [[ "$branch_last_segment" =~ ^[0-9]{3}- ]]; then
        valid="true"
    fi

    # Protected branches (must not operate)
    if [[ "$branch_lower" =~ ^(main|dev)$ ]]; then
        protected="true"
    fi
    if [[ "$branch_lower" =~ ^release-v[0-9]+(\.[0-9]+)*$ ]]; then
        protected="true"
    fi

    if [[ "$branch_lower" =~ ^issue/[0-9]+(-[a-z0-9-]+)?$ ]]; then
        valid="true"
    fi

    if [[ "$branch_lower" =~ ^epic/[0-9]+(-[a-z0-9-]+)?$ ]]; then
        valid="true"
    fi

    if [[ "$protected" == "true" ]]; then
        echo "ERROR: Protected branch detected: $branch. This workflow is disabled on protected branches (main, dev, release-v*)." >&2
        echo "Switch to a feature branch: issue/<ISSUE>[-summary], epic/<ISSUE>[-summary], or use a key ending with NNN-name." >&2
        return 1
    fi

    if [[ "$valid" != "true" ]]; then
        echo "ERROR: Not on a recognized feature branch. Current: $branch (checked: $branch_last_segment)" >&2
        echo "Accepted patterns: issue/<ISSUE>[-summary] | epic/<ISSUE>[-summary] | <...>/NNN-name" >&2
        return 1
    fi

    return 0
}

get_feature_dir() { echo "$1/specs/$2"; }

get_feature_paths() {
    local repo_root=$(get_repo_root)
    local current_branch=$(get_current_branch)
    local has_git_repo="false"

    if has_git; then
        has_git_repo="true"
    fi

    # Resolve feature directory key. Prefer exact match; otherwise fall back to sanitized branch name
    # to align with create-new-feature.sh behavior.
    local feature_key="$current_branch"
    local exact_dir=$(get_feature_dir "$repo_root" "$feature_key")
    if [[ ! -d "$exact_dir" ]]; then
        # Sanitize: lower-case, replace non [a-z0-9] with '-', collapse repeats, trim edges
        local sanitized
        sanitized=$(echo "$current_branch" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//')
        feature_key="$sanitized"
    fi

    local feature_dir=$(get_feature_dir "$repo_root" "$feature_key")

    cat <<EOF
REPO_ROOT='$repo_root'
CURRENT_BRANCH='$current_branch'
HAS_GIT='$has_git_repo'
FEATURE_DIR='$feature_dir'
FEATURE_SPEC='$feature_dir/spec.md'
IMPL_PLAN='$feature_dir/plan.md'
TASKS='$feature_dir/tasks.md'
RESEARCH='$feature_dir/research.md'
DATA_MODEL='$feature_dir/data-model.md'
QUICKSTART='$feature_dir/quickstart.md'
CONTRACTS_DIR='$feature_dir/contracts'
EOF
}

check_file() { [[ -f "$1" ]] && echo "  ✓ $2" || echo "  ✗ $2"; }
check_dir() { [[ -d "$1" && -n $(ls -A "$1" 2>/dev/null) ]] && echo "  ✓ $2" || echo "  ✗ $2"; }
