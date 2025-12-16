## PR Merge

Automatically merge Pull Requests after comprehensive quality verification and approval.

### Usage

```bash
# Auto-merge after quality gates pass
/pr-merge
"Verify all quality gates and automatically merge the current PR"

# Merge specific PR with verification
/pr-merge --pr 123
"Verify and merge PR #123 after all checks pass"

# Dry run to check merge readiness
/pr-merge --dry-run
"Check if PR is ready for merge without actually merging"
```

### Basic Examples

```bash
# Complete quality verification and merge
gh pr checks && gh pr view --json reviewDecision
"Verify CI status and approval status, then proceed with merge if all conditions are met"

# Conditional merge with safety checks
gh pr view --json isDraft,mergeable,reviewDecision
"Check draft status, merge conflicts, and review approvals before merging"
```

### Pre-merge Verification Checklist

#### 1. CI Status Check
```bash
# Verify all CI checks pass
gh pr checks --required
"Ensure all required CI checks are green"
```

#### 2. Review Status Check
```bash
# Check approval status
gh pr view --json reviewDecision,reviews
"Verify PR has required approvals and no pending change requests"
```

#### 3. Branch Status Check
```bash
# Check merge conflicts
gh pr view --json mergeable,mergeableState
"Ensure no merge conflicts exist"
```

#### 4. Draft Status Check
```bash
# Ensure PR is not in draft
gh pr view --json isDraft
"Confirm PR is marked as ready for review"
```

### Merge Strategies

#### Default: Squash and Merge
```bash
# Squash merge (recommended for feature branches)
gh pr merge --squash --delete-branch
"Combine all commits into single commit and delete feature branch"
```

#### Alternative: Merge Commit
```bash
# Create merge commit (for release branches)
gh pr merge --merge --delete-branch
"Create explicit merge commit preserving commit history"
```

#### Alternative: Rebase and Merge
```bash
# Rebase merge (for linear history)
gh pr merge --rebase --delete-branch
"Replay commits on top of base branch"
```

### Safety Mechanisms

#### 1. Required Conditions
- ✅ All required CI checks must pass
- ✅ At least one approval from code owner
- ✅ No pending change requests
- ✅ No merge conflicts
- ✅ PR must not be in draft state
- ✅ Branch protection rules satisfied

#### 2. Optional Quality Gates
- ⚠️ Code coverage threshold met
- ⚠️ Security scan passed
- ⚠️ Performance benchmarks acceptable
- ⚠️ Documentation updated

#### 3. Emergency Bypass (Admin Only)
```bash
# Force merge with admin privileges (use with caution)
/pr-merge --force --admin
"Override protection rules (requires admin permissions)"
```

### Execution Flow

```bash
#!/bin/bash

# 1. Detect current PR
detect_current_pr() {
  local current_branch=$(git branch --show-current)
  gh pr list --head $current_branch --json number --jq '.[0].number'
}

# 2. Comprehensive verification
verify_merge_readiness() {
  local pr_number=$1

  # Check CI status
  local ci_status=$(gh pr checks $pr_number --required --json state --jq '.[] | select(.state != "SUCCESS") | length')
  if [ $ci_status -gt 0 ]; then
    echo "❌ Required CI checks not passing"
    return 1
  fi

  # Check review status
  local review_decision=$(gh pr view $pr_number --json reviewDecision --jq '.reviewDecision')
  if [ "$review_decision" != "APPROVED" ]; then
    echo "❌ PR not approved (status: $review_decision)"
    return 1
  fi

  # Check draft status
  local is_draft=$(gh pr view $pr_number --json isDraft --jq '.isDraft')
  if [ "$is_draft" = "true" ]; then
    echo "❌ PR is still in draft"
    return 1
  fi

  # Check merge conflicts
  local mergeable=$(gh pr view $pr_number --json mergeable --jq '.mergeable')
  if [ "$mergeable" != "MERGEABLE" ]; then
    echo "❌ PR has merge conflicts"
    return 1
  fi

  echo "✅ All merge conditions satisfied"
  return 0
}

# 3. Execute merge
execute_merge() {
  local pr_number=$1
  local strategy=${2:-"squash"}

  case $strategy in
    "squash")
      gh pr merge $pr_number --squash --delete-branch
      ;;
    "merge")
      gh pr merge $pr_number --merge --delete-branch
      ;;
    "rebase")
      gh pr merge $pr_number --rebase --delete-branch
      ;;
    *)
      echo "❌ Unknown merge strategy: $strategy"
      return 1
      ;;
  esac

  echo "✅ PR $pr_number merged successfully"
}

# Main execution
main() {
  local pr_number=$(detect_current_pr)

  if [ -z "$pr_number" ]; then
    echo "❌ No PR found for current branch"
    exit 1
  fi

  echo "🔍 Verifying merge readiness for PR #$pr_number"

  if verify_merge_readiness $pr_number; then
    if [ "$DRY_RUN" = "true" ]; then
      echo "✅ [DRY RUN] PR #$pr_number is ready for merge"
    else
      echo "🚀 Proceeding with merge..."
      execute_merge $pr_number $MERGE_STRATEGY
    fi
  else
    echo "❌ PR #$pr_number is not ready for merge"
    exit 1
  fi
}
```

### Options

- `--pr <number>`: Specify PR number (auto-detect from current branch if omitted)
- `--strategy <squash|merge|rebase>`: Choose merge strategy (default: squash)
- `--dry-run`: Check merge readiness without actually merging
- `--force`: Override protection rules (admin only)
- `--no-delete-branch`: Keep feature branch after merge

### Integration with Existing Workflow

```bash
# Complete automated workflow
/pr-check          # Quality verification
/pr-review         # Comprehensive review
/pr-feedback       # Address any issues
/pr-auto-update    # Update metadata
/pr-merge          # Auto-merge when ready
```

### Common Use Cases

#### 1. Feature Branch Completion
```bash
# After development completion
/pr-merge --strategy squash
"Squash all feature commits into clean single commit"
```

#### 2. Hotfix Deployment
```bash
# Emergency fix that needs immediate merge
/pr-merge --strategy merge
"Preserve exact commit history for audit trail"
```

#### 3. Release Branch Integration
```bash
# Merge release branch with full history
/pr-merge --strategy merge --no-delete-branch
"Integrate release while preserving branch for future reference"
```

### Troubleshooting

#### Common Issues
1. **PR not approved**: Request review from code owners
2. **CI checks failing**: Use `/pr-feedback` to analyze and fix
3. **Merge conflicts**: Resolve conflicts and update branch
4. **Draft status**: Use `gh pr ready` to mark as ready for review
5. **Missing branch protection**: Contact repository admin

#### Error Recovery
```bash
# If merge fails, analyze the issue
gh pr view $PR_NUMBER --json mergeableState,statusCheckRollup
"Check detailed merge status and CI results"

# Fix issues and retry
/pr-feedback && /pr-merge --dry-run
"Address issues and verify before retry"
```

### Security Considerations

1. **Permission Verification**: Ensure user has merge permissions
2. **Branch Protection**: Respect all configured protection rules
3. **Audit Trail**: Log all merge actions with user attribution
4. **Rollback Plan**: Document merge reversal procedures if needed

### Notes

- **Requires GitHub CLI**: `gh` must be authenticated
- **Branch Permissions**: User must have write access to target branch
- **Protection Rules**: All repository protection rules must be satisfied
- **Notification**: Team members are notified via GitHub's standard merge notifications
