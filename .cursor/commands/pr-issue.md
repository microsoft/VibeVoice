## Issue List

Displays a prioritized list of open issues in the current repository.

### Usage

```bash
# Request from Claude
"Show a prioritized list of open issues"
```

### Basic Examples

```bash
# Get repository information
gh repo view --json nameWithOwner | jq -r '.nameWithOwner'

# Get open issue information and request from Claude
gh issue list --state open --json number,title,author,createdAt,updatedAt,labels,assignees,comments --limit 30

"Organize the above issues by priority, including a 2-line summary for each issue. Generate URLs using the repository name obtained above"
```

### Display Format

```
Open Issues List (by Priority)

### High Priority
#number Title [labels] | Author | Time since opened | Comment count | Assignee
      ├─ Summary line 1
      └─ Summary line 2
      https://github.com/owner/repo/issues/number

### Medium Priority
(Similar format)

### Low Priority
(Similar format)
```

### Priority Assessment Criteria

**High Priority**

- Issues with `bug` label
- Issues with `critical` or `urgent` labels
- Issues with `security` label

**Medium Priority**

- Issues with `enhancement` label
- Issues with `feature` label
- Issues with assignees

**Low Priority**

- Issues with `documentation` label
- Issues with `good first issue` label
- Issues with `wontfix` or `duplicate` labels

### Label Filtering

```bash
# Get only issues with specific label
gh issue list --state open --label "bug" --json number,title,author,createdAt,labels,comments --limit 30

# Filter with multiple labels (AND condition)
gh issue list --state open --label "bug,high-priority" --json number,title,author,createdAt,labels,comments --limit 30
```

### Notes

- Requires GitHub CLI (`gh`)
- Only displays issues in open state
- Shows maximum 30 issues
- Elapsed time is from when the issue was opened
- Issue URLs are automatically generated from the actual repository name
