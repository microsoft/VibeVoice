## PR List

Displays a prioritized list of open PRs in the current repository.

### Usage

```bash
# Request from Claude
"Show a prioritized list of open PRs"
```

### Basic Examples

```bash
# Get repository information
gh repo view --json nameWithOwner | jq -r '.nameWithOwner'

# Get open PR information and request from Claude
gh pr list --state open --draft=false --json number,title,author,createdAt,additions,deletions,reviews --limit 30

"Organize the above PRs by priority, including a 2-line summary for each PR. Generate URLs using the repository name obtained above"
```

### Display Format

```
Open PRs List (by Priority)

### High Priority
#number Title [Draft/DNM] | Author | Time since opened | Approved count | +additions/-deletions
      ├─ Summary line 1
      └─ Summary line 2
      https://github.com/owner/repo/pull/number

### Medium Priority
(Similar format)

### Low Priority
(Similar format)
```

### Priority Assessment Criteria

**High Priority**

- `fix:` Bug fixes
- `release:` Release work

**Medium Priority**

- `feat:` New features
- `update:` Feature improvements
- Other regular PRs

**Low Priority**

- PRs containing DO NOT MERGE
- Draft PRs with `test:`, `build:`, `perf:`

### Notes

- Requires GitHub CLI (`gh`)
- Only displays PRs in open state (Drafts are excluded)
- Shows maximum 30 PRs
- Elapsed time is from when the PR was opened
- PR URLs are automatically generated from the actual repository name
