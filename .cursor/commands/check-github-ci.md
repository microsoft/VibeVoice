## GitHub CI Monitoring

Monitors GitHub Actions CI status and tracks until completion.

### Usage

```bash
# Check CI status
gh pr checks
```

### Basic Examples

```bash
# Check CI after creating PR
gh pr create --title "Add new feature" --body "Description"
gh pr checks
```

### Collaboration with Claude

```bash
# Flow from CI check to correction
gh pr checks
"Analyze CI check results and suggest fixes if there are failures"

# Recheck after correction
git push origin feature-branch
gh pr checks
"Check CI results after correction to confirm no issues"
```

### Example Execution Results

```
All checks were successful
0 cancelled, 0 failing, 8 successful, 3 skipped, and 0 pending checks

   NAME                                    DESCRIPTION                ELAPSED  URL
○  Build/test (pull_request)                                          5m20s    https://github.com/user/repo/actions/runs/123456789
○  Build/lint (pull_request)                                          2m15s    https://github.com/user/repo/actions/runs/123456789
○  Security/scan (pull_request)                                       1m30s    https://github.com/user/repo/actions/runs/123456789
○  Type Check (pull_request)                                          45s      https://github.com/user/repo/actions/runs/123456789
○  Commit Messages (pull_request)                                     12s      https://github.com/user/repo/actions/runs/123456789
-  Deploy Preview (pull_request)                                               https://github.com/user/repo/actions/runs/123456789
-  Visual Test (pull_request)                                                  https://github.com/user/repo/actions/runs/123456789
```

### Notes

- Check details when failed
- Wait for all checks to complete before merging
- Re-run `gh pr checks` as needed
