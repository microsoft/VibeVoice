## PR Create

Creates Pull Requests automatically by analyzing your Git changes for a smoother workflow.

### Usage

```bash
# Auto-create PR from your changes
git add . && git commit -m "feat: Implement user authentication"
"Create a Draft PR with the right description and labels"

# Keep your existing template
cp .github/PULL_REQUEST_TEMPLATE.md pr_body.md
"Fill in the blanks but keep the template structure intact"

# Mark as ready when done
gh pr ready
"Switch to Ready for Review after checking quality"
```

### Basic Examples

```bash
# 1. Create branch and commit
git checkout main && git pull
git checkout -b feat-user-profile
git add . && git commit -m "feat: Implement user profile feature"
git push -u origin feat-user-profile

# 2. Create PR
"Please create a PR:
1. Check what changed with git diff --cached
2. Use the PR template from .github/PULL_REQUEST_TEMPLATE.md
3. Pick up to 3 labels that match the changes
4. Create it as a Draft (keep HTML comments)"

# 3. Make it ready after CI passes
"Once CI is green, mark the PR as Ready for Review"
```

### Execution Steps

#### 1. Create Branch

```bash
# Branch naming: {type}-{subject}
git checkout main
git pull
git checkout -b feat-user-authentication

# Confirm you're on the right branch
git branch --show-current
```

#### 2. Commit

```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "feat: Implement user authentication API"
```

#### 3. Push to Remote

```bash
# First push (sets upstream)
git push -u origin feat-user-authentication

# Later pushes
git push
```

#### 4. Create Draft PR with Automatic Analysis

**Step 1: Analyze Changes**

```bash
# See what files changed
git diff --cached --name-only

# Review the actual changes (first 1000 lines)
git diff --cached | head -1000
```

**Step 2: Auto-generate Description**

```bash
# Template priority:
# 1. Keep existing PR description as-is
# 2. Use .github/PULL_REQUEST_TEMPLATE.md
# 3. Fall back to default template

cp .github/PULL_REQUEST_TEMPLATE.md pr_body.md
# Fill empty sections only - don't touch HTML comments or separators
```

**Step 3: Auto-select Labels**

```bash
# Get available labels (non-interactive)
"Retrieve available labels from .github/labels.yml or GitHub repository and automatically select appropriate labels based on changes"

# Auto-selection by pattern matching (max 3)
# - Documentation: *.md, docs/ → documentation|docs
# - Tests: test, spec → test|testing
# - Bug fixes: fix|bug → bug|fix
# - New features: feat|feature → feature|enhancement
```

**Step 4: Create PR via GitHub API (Preserve HTML Comments)**

```bash
# Create PR
"Create a Draft PR with the following information:
- Title: Auto-generated from commit message
- Description: Properly filled using .github/PULL_REQUEST_TEMPLATE.md
- Labels: Auto-selected from changes (max 3)
- Base branch: main
- Preserve all HTML comments"
```

**Method B: GitHub MCP (Fallback)**

```javascript
// Create PR while preserving HTML comments
mcp_github_create_pull_request({
  owner: 'organization',
  repo: 'repository',
  base: 'main',
  head: 'feat-user-authentication',
  title: 'feat: Implement user authentication',
  body: prBodyContent, // Full content including HTML comments
  draft: true,
  maintainer_can_modify: true,
});
```

### Auto Label Selection System

#### Determining from File Patterns

- **Documentation**: `*.md`, `README`, `docs/` → `documentation|docs|doc`
- **Tests**: `test`, `spec` → `test|testing`
- **CI/CD**: `.github/`, `*.yml`, `Dockerfile` → `ci|build|infra|ops`
- **Dependencies**: `package.json`, `pubspec.yaml` → `dependencies|deps`

#### Determining from Content

- **Bug fixes**: `fix|bug|error|crash|repair` → `bug|fix`
- **New features**: `feat|feature|add|implement|new-feature|implementation` → `feature|enhancement|feat`
- **Refactoring**: `refactor|clean|restructure` → `refactor|cleanup|clean`
- **Performance**: `performance|perf|optimize` → `performance|perf`
- **Security**: `security|secure` → `security`

#### Constraints

- **Max 3 labels**: Upper limit for automatic selection
- **Existing labels only**: Prohibited from creating new labels
- **Partial match**: Determined by keyword inclusion in label names

### Project Guidelines

#### Basic Approach

1. **Always start as Draft**: All PRs must be created in Draft state
2. **Gradual quality improvement**: Phase 1 (Basic implementation) → Phase 2 (Add tests) → Phase 3 (Update documentation)
3. **Appropriate labels**: Always add up to 3 labels
4. **Use templates**: Always use `.github/PULL_REQUEST_TEMPLATE.md`
5. **Japanese spacing**: Always add half-width space between Japanese text and alphanumerics

#### Branch Naming Convention

```text
{type}-{subject}

Examples:
- feat-user-profile
- fix-login-error
- refactor-api-client
```

#### Commit Messages

```text
{type}: {description}

Examples:
- feat: Implement user authentication API
- fix: Correct login error
- docs: Update README
```

### Template Processing System

#### Processing Priority

1. **Existing PR description**: Keep everything that's already written
2. **Project template**: Use `.github/PULL_REQUEST_TEMPLATE.md`
3. **Default template**: Use this if nothing else exists

#### Existing Content Preservation Rules

- **Don't touch existing content**: Leave what's already there alone
- **Fill in the blanks only**: Add content where it's missing
- **Keep functional comments**: Like `<!-- Copilot review rule -->`
- **Keep HTML comments**: All `<!-- ... -->` stay as-is
- **Keep separators**: Things like `---` stay put

#### Handling HTML Comment Preservation

**Heads up**: GitHub CLI (`gh pr edit`) escapes HTML comments, and shell processing can mess things up with strings like `EOF < /dev/null`.

**How to fix this**:

1. **Use GitHub API's --field option**: This handles escaping properly
2. **Keep it simple**: Skip complex pipes and redirects
3. **Don't remove anything**: Keep all HTML comments and templates intact

### Review Comment Responses

```bash
# Commit your fixes
git add .
git commit -m "fix: Address review feedback"
git push
```

### Notes

#### Importance of HTML Comment Preservation

- **GitHub CLI issue**: `gh pr edit` escapes HTML comments and can break things
- **The fix**: Use GitHub API's `--field` option for proper handling
- **Keep everything**: Don't remove HTML comments - keep the whole template

#### Automation Constraints

- **No new labels**: Can only use labels from `.github/labels.yml`
- **3 labels max**: That's the limit for auto-selection
- **Hands off manual content**: Never change what someone wrote

#### Step-by-Step Quality

- **Start with Draft**: Every PR begins as a draft
- **Check CI**: Run `gh pr checks` to see the status
- **Mark as ready**: Use `gh pr ready` when quality looks good
- **Follow the template**: Stick to your project's structure
