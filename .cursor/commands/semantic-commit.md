## Semantic Commit

Breaks big changes into small, meaningful commits with proper messages. Uses only standard git commands.

### Usage

```bash
/semantic-commit [options]
```

### Options

- `--dry-run`: Show proposed commit splits without actually committing
- `--lang <language>`: Force language for commit messages (en, ja)
- `--max-commits <number>`: Specify maximum number of commits (default: 10)

### Basic Examples

```bash
# Analyze current changes and commit in logical units
/semantic-commit

# Check split proposal only (no actual commit)
/semantic-commit --dry-run

# Generate commit messages in English
/semantic-commit --lang en

# Split into maximum 5 commits
/semantic-commit --max-commits 5
```

### How It Works

1. **Analyze Changes**: Check what changed with `git diff HEAD`
2. **Group Files**: Put related files together
3. **Create Messages**: Write semantic commit messages for each group
4. **Commit Step by Step**: Commit each group after you approve

### When to Split Changes

#### What Makes a Change "Large"

We split when we see:

1. **Many Files**: 5+ files changed
2. **Many Lines**: 100+ lines changed
3. **Multiple Features**: Changes in 2+ areas
4. **Mixed Types**: feat + fix + docs together

```bash
# Analyze change scale
CHANGED_FILES=$(git diff HEAD --name-only | wc -l)
CHANGED_LINES=$(git diff HEAD --stat | tail -1 | grep -o '[0-9]\+ insertions\|[0-9]\+ deletions' | awk '{sum+=$1} END {print sum}')

if [ $CHANGED_FILES -ge 5 ] || [ $CHANGED_LINES -ge 100 ]; then
  echo "Large change detected: splitting recommended"
fi
```

#### How to Split into Small, Meaningful Commits

##### 1. Splitting by Functional Boundaries

```bash
# Identify functional units from directory structure
git diff HEAD --name-only | cut -d'/' -f1-2 | sort | uniq
# → src/auth, src/api, components/ui, etc.
```

##### 2. Separation by Change Type

```bash
# New files vs existing file modifications
git diff HEAD --name-status | grep '^A' # New files
git diff HEAD --name-status | grep '^M' # Modified files
git diff HEAD --name-status | grep '^D' # Deleted files
```

##### 3. Dependency Analysis

```bash
# Detect import relationship changes
git diff HEAD | grep -E '^[+-].*import|^[+-].*require' | \
cut -d' ' -f2- | sort | uniq
```

#### Detailed File Analysis

```bash
# Get list of changed files
git diff HEAD --name-only

# Analyze each file's changes individually
git diff HEAD -- <file>

# Determine change type for files
git diff HEAD --name-status | while read status file; do
  case $status in
    A) echo "$file: New creation" ;;
    M) echo "$file: Modification" ;;
    D) echo "$file: Deletion" ;;
    R*) echo "$file: Renamed" ;;
  esac
done
```

#### How to Group Files

1. **By Feature**: Keep related functions together
   - `src/auth/` files → Authentication
   - `components/` files → UI components

2. **By Type**: Same kind of changes
   - Only tests → `test:`
   - Only docs → `docs:`
   - Only config → `chore:`

3. **By Dependencies**: Files that need each other
   - Model + Migration
   - Component + Style

4. **By Size**: Keep commits manageable
   - Max 10 files per commit
   - Keep related files together

### Output Example

```bash
$ /semantic-commit

Analyzing changes...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detected changes:
• src/auth/login.ts (modified)
• src/auth/register.ts (new)
• src/auth/types.ts (modified)
• tests/auth.test.ts (new)
• docs/authentication.md (new)

Proposed commit splits:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Commit 1/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Message: feat: implement user registration and login system
Included files:
  • src/auth/login.ts
  • src/auth/register.ts
  • src/auth/types.ts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Commit 2/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Message: test: add comprehensive tests for authentication system
Included files:
  • tests/auth.test.ts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Commit 3/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Message: docs: add authentication system documentation
Included files:
  • docs/authentication.md

Execute commit with this split plan? (y/n/edit):
```

### Your Options

- `y`: Go with the proposed split
- `n`: Cancel everything
- `edit`: Change commit messages
- `merge <number1> <number2>`: Combine commits
- `split <number>`: Break up a commit more

### Dry Run Mode

```bash
$ /semantic-commit --dry-run

Analyzing changes... (DRY RUN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Commit split proposal display]

ℹ️  DRY RUN mode: No actual commits will be executed
💡 To execute, run again without --dry-run option
```

### Smart Features

#### 1. Understands Your Project

- Detects project type from config files
- Figures out features from folder structure

#### 2. Change Pattern Recognition

```bash
# Detect bug fix patterns
- Keywords like "fix", "bug", "error"
- Addition of exception handling
- Condition branch fixes

# Detect new feature patterns
- New file creation
- New method additions
- API endpoint additions
```

#### 3. Dependency Analysis

- Changes to import statements
- Addition/modification of type definitions
- Relationship with configuration files

### How It's Built

#### Step-by-Step Commits with Git

##### 1. Preprocessing: Save Current State

```bash
# Reset unstaged changes if any
git reset HEAD
git status --porcelain > /tmp/original_state.txt

# Check working branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Working branch: $CURRENT_BRANCH"
```

##### 2. Sequential Commit Execution by Group

```bash
# Read split plan
while IFS= read -r commit_plan; do
  group_num=$(echo "$commit_plan" | cut -d':' -f1)
  files=$(echo "$commit_plan" | cut -d':' -f2- | tr ' ' '\n')

  echo "=== Executing commit $group_num ==="

  # Stage only relevant files
  echo "$files" | while read file; do
    if [ -f "$file" ]; then
      git add "$file"
      echo "Staged: $file"
    fi
  done

  # Check staging status
  staged_files=$(git diff --staged --name-only)
  if [ -z "$staged_files" ]; then
    echo "Warning: No files staged"
    continue
  fi

  # Generate commit message (LLM analysis)
  commit_msg=$(generate_commit_message_for_staged_files)

  # User confirmation
  echo "Proposed commit message: $commit_msg"
  echo "Staged files:"
  echo "$staged_files"
  read -p "Execute this commit? (y/n): " confirm

  if [ "$confirm" = "y" ]; then
    # Execute commit
    git commit -m "$commit_msg"
    echo "✅ Commit $group_num completed"
  else
    # Cancel staging
    git reset HEAD
    echo "❌ Skipped commit $group_num"
  fi

done < /tmp/commit_plan.txt
```

##### 3. Error Handling and Rollback

```bash
# Handle pre-commit hook failures
commit_with_retry() {
  local commit_msg="$1"
  local max_retries=2
  local retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    if git commit -m "$commit_msg"; then
      echo "✅ Commit successful"
      return 0
    else
      echo "❌ Commit failed (attempt $((retry_count + 1))/$max_retries)"

      # Incorporate automatic fixes from pre-commit hooks
      if git diff --staged --quiet; then
        echo "Changes automatically fixed by pre-commit hook"
        git add -u
      fi

      retry_count=$((retry_count + 1))
    fi
  done

  echo "❌ Failed to commit. Please check manually."
  return 1
}

# Recover from interruptions
resume_from_failure() {
  echo "Detected interrupted commit process"
  echo "Current staging status:"
  git status --porcelain

  read -p "Continue processing? (y/n): " resume
  if [ "$resume" = "y" ]; then
    # Resume from last commit
    last_commit=$(git log --oneline -1 --pretty=format:"%s")
    echo "Last commit: $last_commit"
  else
    # Full reset
    git reset HEAD
    echo "Process reset"
  fi
}
```

##### 4. Post-Completion Verification

```bash
# Verify all changes committed
remaining_changes=$(git status --porcelain | wc -l)
if [ $remaining_changes -eq 0 ]; then
  echo "✅ All changes committed"
else
  echo "⚠️  Uncommitted changes remain:"
  git status --short
fi

# Display commit history
echo "Created commits:"
git log --oneline -n 10 --graph
```

##### 5. Suppress Automatic Push

```bash
# Note: No automatic push
echo "📝 Note: Automatic push not performed"
echo "If needed, push with the following command:"
echo "  git push origin $CURRENT_BRANCH"
```

#### Split Algorithm Details

##### Step 1: Initial Analysis

```bash
# Get and classify all changed files
git diff HEAD --name-status | while read status file; do
  echo "$status:$file"
done > /tmp/changes.txt

# Statistics of changes by functional directory
git diff HEAD --name-only | cut -d'/' -f1-2 | sort | uniq -c
```

##### Step 2: Initial Grouping by Functional Boundaries

```bash
# Directory-based grouping
GROUPS=$(git diff HEAD --name-only | cut -d'/' -f1-2 | sort | uniq)
for group in $GROUPS; do
  echo "=== Group: $group ==="
  git diff HEAD --name-only | grep "^$group" | head -10
done
```

##### Step 3: Analyzing Change Similarity

```bash
# Analyze change type for each file
git diff HEAD --name-only | while read file; do
  # Detect new function/class additions
  NEW_FUNCTIONS=$(git diff HEAD -- "$file" | grep -c '^+.*function\|^+.*class\|^+.*def ')

  # Detect bug fix patterns
  BUG_FIXES=$(git diff HEAD -- "$file" | grep -c '^+.*fix\|^+.*bug\|^-.*error')

  # Determine if test file
  if [[ "$file" =~ test|spec ]]; then
    echo "$file: TEST"
  elif [ $NEW_FUNCTIONS -gt 0 ]; then
    echo "$file: FEAT"
  elif [ $BUG_FIXES -gt 0 ]; then
    echo "$file: FIX"
  else
    echo "$file: REFACTOR"
  fi
done
```

##### Step 4: Dependency-based Adjustments

```bash
# Analyze import relationships
git diff HEAD | grep -E '^[+-].*import|^[+-].*from.*import' | \
while read line; do
  echo "$line" | sed 's/^[+-]//' | awk '{print $2}'
done | sort | uniq > /tmp/imports.txt

# Group related files
git diff HEAD --name-only | while read file; do
  basename=$(basename "$file" .js .ts .py)
  related=$(git diff HEAD --name-only | grep "$basename" | grep -v "^$file$")
  if [ -n "$related" ]; then
    echo "Related files: $file <-> $related"
  fi
done
```

##### Step 5: Commit Size Optimization

```bash
# Adjust group size
MAX_FILES_PER_COMMIT=8
current_group=1
file_count=0

git diff HEAD --name-only | while read file; do
  if [ $file_count -ge $MAX_FILES_PER_COMMIT ]; then
    current_group=$((current_group + 1))
    file_count=0
  fi
  echo "Commit $current_group: $file"
  file_count=$((file_count + 1))
done
```

##### Step 6: Determining Final Groups

```bash
# Verify split results
for group in $(seq 1 $current_group); do
  files=$(grep "Commit $group:" /tmp/commit_plan.txt | cut -d':' -f2-)
  lines=$(echo "$files" | xargs git diff HEAD -- | wc -l)
  echo "Commit $group: $(echo "$files" | wc -w) files, $lines lines changed"
done
```

### Conventional Commits Compliance

#### Basic Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Standard Types

**Required Types**:

- `feat`: New feature (user-visible feature addition)
- `fix`: Bug fix

**Optional Types**:

- `build`: Changes to build system or external dependencies
- `chore`: Other changes (no impact on release)
- `ci`: Changes to CI configuration files or scripts
- `docs`: Documentation-only changes
- `style`: Changes that do not affect code meaning (whitespace, formatting, semicolons, etc.)
- `refactor`: Code changes without bug fixes or feature additions
- `perf`: Performance improvements
- `test`: Adding or modifying tests

#### Scope (Optional)

Indicates the affected area of the change:

```
feat(api): add user authentication endpoint
fix(ui): resolve button alignment issue
docs(readme): update installation instructions
```

#### Breaking Change

When there are breaking API changes:

```
feat!: change user API response format

```

or

```
feat(api)!: change authentication flow
```

#### Automatically Detecting Project Conventions

**Important**: If project-specific conventions exist, they take precedence.

##### 1. Check CommitLint Configuration

Automatically detect configuration from the following files:

- `commitlint.config.js`
- `commitlint.config.mjs`
- `commitlint.config.cjs`
- `commitlint.config.ts`
- `.commitlintrc.js`
- `.commitlintrc.json`
- `.commitlintrc.yml`
- `.commitlintrc.yaml`
- `commitlint` section in `package.json`

```bash
# Check example configuration files
cat commitlint.config.mjs
cat .commitlintrc.json
grep -A 10 '"commitlint"' package.json
```

##### 2. Detecting Custom Types

Example of project-specific types:

```javascript
// commitlint.config.mjs
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore',
        'wip',      // Work in progress
        'hotfix',   // Emergency fix
        'release',  // Release
        'deps',     // Dependency update
        'config'    // Configuration change
      ]
    ]
  }
}
```

##### 3. Detecting Language Settings

```javascript
// When project uses Japanese messages
export default {
  rules: {
    'subject-case': [0],  // Disable for Japanese support
    'subject-max-length': [2, 'always', 72]  // Adjust character limit for Japanese
  }
}
```

#### Automatic Analysis Flow

1. **Configuration File Search**

   ```bash
   find . -name "commitlint.config.*" -o -name ".commitlintrc.*" | head -1
   ```

2. **Existing Commit Analysis**

   ```bash
   git log --oneline -50 --pretty=format:"%s"
   ```

3. **Type Usage Statistics**

   ```bash
   git log --oneline -100 --pretty=format:"%s" | \
   grep -oE '^[a-z]+(\([^)]+\))?' | \
   sort | uniq -c | sort -nr
   ```

#### Examples of Project Conventions

##### Angular Style

```
feat(scope): add new feature
fix(scope): fix bug
docs(scope): update documentation
```

##### Gitmoji Combined Style

```
✨ feat: add user registration
🐛 fix: resolve login issue
📚 docs: update API docs
```

##### Japanese Projects

```
feat: add user registration functionality
fix: resolve login process bug
docs: update API documentation
```

### Language Detection

How we figure out your language:

1. **Check CommitLint Settings** for language configuration

   ```bash
   # Determine Japanese if subject-case rule is disabled
   grep -E '"subject-case".*\[0\]|subject-case.*0' commitlint.config.*
   ```

2. **Git log analysis** for automatic determination

   ```bash
   # Analyze language of last 20 commits
   git log --oneline -20 --pretty=format:"%s" | \
   grep -E '^[\x{3040}-\x{30ff}]|[\x{4e00}-\x{9fff}]' | wc -l
   # Japanese mode if over 50% are Japanese
   ```

3. **Project files** language settings

   ```bash
   # Check README.md language
   head -10 README.md | grep -E '^[\x{3040}-\x{30ff}]|[\x{4e00}-\x{9fff}]' | wc -l

   # Check package.json description
   grep -E '"description".*[\x{3040}-\x{30ff}]|[\x{4e00}-\x{9fff}]' package.json
   ```

4. **Comments and strings** analysis in changed files

   ```bash
   # Check comment language in changed files
   git diff HEAD | grep -E '^[+-].*//.*[\x{3040}-\x{30ff}]|[\x{4e00}-\x{9fff}]' | wc -l
   ```

#### Determination Algorithm

```bash
# Calculate language score
JAPANESE_SCORE=0

# 1. CommitLint settings (+3 points)
if grep -q '"subject-case".*\[0\]' commitlint.config.* 2>/dev/null; then
  JAPANESE_SCORE=$((JAPANESE_SCORE + 3))
fi

# 2. Git log analysis (max +2 points)
JAPANESE_COMMITS=$(git log --oneline -20 --pretty=format:"%s" | \
  grep -cE '[\x{3040}-\x{30ff}]|[\x{4e00}-\x{9fff}]' 2>/dev/null || echo 0)
if [ $JAPANESE_COMMITS -gt 10 ]; then
  JAPANESE_SCORE=$((JAPANESE_SCORE + 2))
elif [ $JAPANESE_COMMITS -gt 5 ]; then
  JAPANESE_SCORE=$((JAPANESE_SCORE + 1))
fi

# 3. README.md check (+1 point)
if head -5 README.md 2>/dev/null | grep -qE '[\x{3040}-\x{30ff}]|[\x{4e00}-\x{9fff}]'; then
  JAPANESE_SCORE=$((JAPANESE_SCORE + 1))
fi

# 4. Changed files content check (+1 point)
if git diff HEAD 2>/dev/null | grep -qE '^[+-].*[\x{3040}-\x{30ff}]|[\x{4e00}-\x{9fff}]'; then
  JAPANESE_SCORE=$((JAPANESE_SCORE + 1))
fi

# Determine: Japanese mode if score >= 3
if [ $JAPANESE_SCORE -ge 3 ]; then
  LANGUAGE="ja"
else
  LANGUAGE="en"
fi
```

### Auto-Loading Config

#### What Happens at Runtime

We check for config files in this order:

1. **Search for CommitLint configuration files**

   ```bash
   # Search in this order, use first found file
   commitlint.config.mjs
   commitlint.config.js
   commitlint.config.cjs
   commitlint.config.ts
   .commitlintrc.js
   .commitlintrc.json
   .commitlintrc.yml
   .commitlintrc.yaml
   package.json (commitlint section)
   ```

2. **Parse configuration content**
   - Extract list of available types
   - Check for scope restrictions
   - Get message length limits
   - Check language settings

3. **Analyze existing commit history**

   ```bash
   # Learn usage patterns from recent commits
   git log --oneline -100 --pretty=format:"%s" | \
   head -20
   ```

#### Analyzing Configuration Examples

**Standard commitlint.config.mjs**:

```javascript
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore']
    ],
    'scope-enum': [
      2,
      'always',
      ['api', 'ui', 'core', 'auth', 'db']
    ]
  }
}
```

**Japanese-compatible configuration**:

```javascript
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'subject-case': [0],  // Disable for Japanese
    'subject-max-length': [2, 'always', 72],
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore']
    ]
  }
}
```

**Configuration with custom types**:

```javascript
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore',
        'wip',      // Work in Progress
        'hotfix',   // Emergency fix
        'release',  // Release preparation
        'deps',     // Dependency update
        'config'    // Configuration change
      ]
    ]
  }
}
```

#### Fallback Behavior

If no configuration file is found:

1. **Automatic inference** through git log analysis

   ```bash
   # Extract types from last 100 commits
   git log --oneline -100 --pretty=format:"%s" | \
   grep -oE '^[a-z]+(\([^)]+\))?' | \
   sort | uniq -c | sort -nr
   ```

2. **Default to Conventional Commits standard**

   ```
   feat, fix, docs, style, refactor, perf, test, chore, build, ci
   ```

3. **Language determination**
   - Japanese mode if over 50% of commits are in Japanese
   - English mode otherwise

### Requirements

- Must be in a Git repo
- Need uncommitted changes
- Staged changes get reset temporarily

### Important

- **Won't push**: You need to `git push` yourself
- **Same branch**: Commits stay in current branch
- **Back up first**: Consider `git stash` for safety

### Which Rules Win

When making commit messages, we follow this order:

1. **CommitLint settings** (highest priority)
   - Settings in `commitlint.config.*` files
   - Custom types and scope restrictions
   - Message length and case restrictions

2. **Existing commit history** (second priority)
   - Statistics of actually used types
   - Message language (Japanese/English)
   - Scope usage patterns

3. **Project type** (third priority)
   - `package.json` → Node.js project
   - `Cargo.toml` → Rust project
   - `pom.xml` → Java project

4. **Conventional Commits standard** (fallback)
   - Standard behavior when no settings found

#### Examples of Convention Detection

**Automatic scope detection in Monorepo**:

```bash
# Infer scopes from packages/ folder
ls packages/ | head -10
# → Propose api, ui, core, auth, etc. as scopes
```

**Framework-specific conventions**:

```javascript
// For Angular projects
{
  'scope-enum': [2, 'always', [
    'animations', 'common', 'core', 'forms', 'http', 'platform-browser',
    'platform-server', 'router', 'service-worker', 'upgrade'
  ]]
}

// For React projects
{
  'scope-enum': [2, 'always', [
    'components', 'hooks', 'utils', 'types', 'styles', 'api'
  ]]
}
```

**Company/team-specific conventions**:

```javascript
// Common pattern in Japanese companies
{
  'type-enum': [2, 'always', [
    'feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore',
    'wip',      // Work in progress (for pull requests)
    'hotfix',   // Emergency fix
    'release'   // Release preparation
  ]],
  'subject-case': [0],  // Support Japanese
  'subject-max-length': [2, 'always', 72]  // Longer limit for Japanese
}
```

### Best Practices

1. **Follow the rules**: Use existing patterns
2. **Keep it small**: One logical change per commit
3. **Be clear**: Say what changed
4. **Group smart**: Related files together
5. **Tests separate**: Test commits on their own
6. **Use configs**: CommitLint helps teams stay consistent

### Real-world Split Examples (Before/After)

#### Example 1: Large Authentication System Addition

**Before (one massive commit):**

```bash
# Changed files (15 files, 850 lines changed)
src/auth/login.js          # New
src/auth/register.js       # New
src/auth/password.js       # New
src/auth/types.js          # New
src/api/auth-routes.js     # New
src/middleware/auth.js     # New
src/database/migrations/001_users.sql  # New
src/database/models/user.js            # New
tests/auth/login.test.js   # New
tests/auth/register.test.js # New
tests/api/auth-routes.test.js # New
docs/authentication.md    # New
package.json              # Dependency addition
README.md                 # Usage addition
.env.example             # Environment variable example

# Problematic conventional commit
feat: implement complete user authentication system with login, registration, password reset, API routes, database models, tests and documentation
```

**After (split into 5 meaningful commits):**

```bash
# Commit 1: Database foundation
feat(db): add user model and authentication schema

Included files:
- src/database/migrations/001_users.sql
- src/database/models/user.js
- src/auth/types.js

Reason: Database structure is the foundation for other features, committed first

# Commit 2: Authentication logic
feat(auth): implement core authentication functionality

Included files:
- src/auth/login.js
- src/auth/register.js
- src/auth/password.js
- src/middleware/auth.js

Reason: Core authentication business logic committed together

# Commit 3: API endpoints
feat(api): add authentication API routes

Included files:
- src/api/auth-routes.js

Reason: API layer depends on authentication logic, committed later

# Commit 4: Comprehensive tests
test(auth): add comprehensive authentication tests

Included files:
- tests/auth/login.test.js
- tests/auth/register.test.js
- tests/api/auth-routes.test.js

Reason: Tests added after implementation is complete

# Commit 5: Documentation and configuration
docs(auth): add authentication documentation and configuration

Included files:
- docs/authentication.md
- package.json
- README.md
- .env.example

Reason: Documentation and configuration committed together at the end
```

#### Example 2: Mixed Bug Fixes and Refactoring

**Before (problematic mixed commit):**

```bash
# Changed files (8 files, 320 lines changed)
src/user/service.js       # Bug fixes + Refactoring
src/user/validator.js     # New (refactoring)
src/auth/middleware.js    # Bug fix
src/api/user-routes.js    # Bug fix + Error handling improvement
tests/user.test.js        # Test addition
tests/auth.test.js        # Bug fix test addition
docs/user-api.md          # Documentation update
package.json              # Dependency update

# Problematic commit
fix: resolve user validation bugs and refactor validation logic with improved error handling
```

**After (split into 3 commits by type):**

```bash
# Commit 1: Critical bug fixes
fix: resolve user validation and authentication bugs

Included files:
- src/user/service.js (bug fix portion only)
- src/auth/middleware.js
- tests/auth.test.js (bug fix tests only)

Reason: Production-affecting bugs fixed with highest priority

# Commit 2: Validation logic refactoring
refactor: extract and improve user validation logic

Included files:
- src/user/service.js (refactoring portion)
- src/user/validator.js
- src/api/user-routes.js
- tests/user.test.js

Reason: Structural improvements committed as functional units

# Commit 3: Documentation and dependency update
chore: update documentation and dependencies

Included files:
- docs/user-api.md
- package.json

Reason: Development environment improvements committed together at the end
```

#### Example 3: Simultaneous Development of Multiple Features

**Before (cross-functional massive commit):**

```bash
# Changed files (12 files, 600 lines changed)
src/user/profile.js       # New feature A
src/user/avatar.js        # New feature A
src/notification/email.js # New feature B
src/notification/sms.js   # New feature B
src/api/profile-routes.js # New feature A API
src/api/notification-routes.js # New feature B API
src/dashboard/widgets.js  # New feature C
src/dashboard/charts.js   # New feature C
tests/profile.test.js     # New feature A tests
tests/notification.test.js # New feature B tests
tests/dashboard.test.js   # New feature C tests
package.json              # All features' dependencies

# Problematic commit
feat: add user profile management, notification system and dashboard widgets
```

**After (split into 4 commits by feature):**

```bash
# Commit 1: User profile feature
feat(profile): add user profile management

Included files:
- src/user/profile.js
- src/user/avatar.js
- src/api/profile-routes.js
- tests/profile.test.js

Reason: Profile feature is an independent functional unit

# Commit 2: Notification system
feat(notification): implement email and SMS notifications

Included files:
- src/notification/email.js
- src/notification/sms.js
- src/api/notification-routes.js
- tests/notification.test.js

Reason: Notification feature is an independent functional unit

# Commit 3: Dashboard widgets
feat(dashboard): add interactive widgets and charts

Included files:
- src/dashboard/widgets.js
- src/dashboard/charts.js
- tests/dashboard.test.js

Reason: Dashboard feature is an independent functional unit

# Commit 4: Dependencies and infrastructure update
chore: update dependencies for new features

Included files:
- package.json

Reason: Common dependency updates committed together at the end
```

### Comparison of Splitting Effects

| Item | Before (Massive Commit) | After (Proper Splitting) |
|------|---------------------|-------------------|
| **Reviewability** | ❌ Very difficult | ✅ Each commit is small and reviewable |
| **Bug Tracking** | ❌ Difficult to identify problem location | ✅ Problematic commits can be immediately identified |
| **Reverting** | ❌ Need to revert everything | ✅ Can pinpoint and revert only problematic parts |
| **Parallel Development** | ❌ Conflict-prone | ✅ Feature-based merging is easy |
| **Deployment** | ❌ All features deployed at once | ✅ Staged deployment possible |

### Troubleshooting

#### When Commit Fails

- Check pre-commit hooks
- Resolve dependencies
- Retry with individual files

#### When Splitting is Inappropriate

- Adjust with `--max-commits` option
- Use manual `edit` mode
- Re-run with finer granularity
