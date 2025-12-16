## Commit Message

Generates commit messages from staged changes (git diff --staged). This command only creates messages and copies them to your clipboard—it doesn't run any git commands.

### Usage

```bash
/commit-message [options]
```

### Options

- `--format <format>` : Choose message format (conventional, gitmoji, angular)
- `--lang <language>` : Set language explicitly (en, ja)
- `--breaking` : Include breaking change detection

### Basic Examples

```bash
# Generate message from staged changes (language auto-detected)
# The top suggestion is automatically copied to your clipboard
/commit-message

# Specify language explicitly
/commit-message --lang ja
/commit-message --lang en

# Include breaking change detection
/commit-message --breaking
```

### Prerequisites

**Important**: This command only works with staged changes. Run `git add` first to stage your changes.

```bash
# If nothing is staged, you'll see:
$ /commit-message
No staged changes found. Please run git add first.
```

### Automatic Clipboard Feature

The top suggestion gets copied to your clipboard as a complete command: `git commit -m "message"`. Just paste and run it in your terminal.

**Implementation Notes**:

- Run `pbcopy` in a separate process from the message output
- Use `printf` instead of `echo` to avoid unwanted newlines

### Automatic Project Convention Detection

**Important**: If project-specific conventions exist, they take priority.

#### 1. CommitLint Configuration Check

Automatically detects settings from the following files:

- `commitlint.config.js`
- `commitlint.config.mjs`
- `commitlint.config.cjs`
- `commitlint.config.ts`
- `.commitlintrc.js`
- `.commitlintrc.json`
- `.commitlintrc.yml`
- `.commitlintrc.yaml`
- `package.json` with `commitlint` section

```bash
# Search for configuration files
find . -name "commitlint.config.*" -o -name ".commitlintrc.*" | head -1
```

#### 2. Custom Type Detection

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
        'wip',      // work in progress
        'hotfix',   // urgent fix
        'release',  // release
        'deps',     // dependency update
        'config'    // configuration change
      ]
    ]
  }
}
```

#### 3. Detecting Language Settings

```javascript
// When project uses Japanese messages
export default {
  rules: {
    'subject-case': [0],  // Disabled for Japanese support
    'subject-max-length': [2, 'always', 72]  // Adjusted character limit for Japanese
  }
}
```

#### 4. Existing Commit History Analysis

```bash
# Learn patterns from recent commits
git log --oneline -50 --pretty=format:"%s"

# Type usage statistics
git log --oneline -100 --pretty=format:"%s" | \
grep -oE '^[a-z]+(\([^)]+\))?' | \
sort | uniq -c | sort -nr
```

### Automatic Language Detection

Automatically switches between Japanese/English based on:

1. **CommitLint configuration** language settings
2. **git log analysis** automatic detection
3. **Project file** language settings
4. **Changed file** comment and string analysis

Default is English. Generates in Japanese if detected as Japanese project.

### Message Format

#### Conventional Commits (Default)

```
<type>: <description>
```

**Important**: Always generates single-line commit messages. Does not generate multi-line messages.

**Note**: Project-specific conventions take priority if they exist.

### Standard Types

**Required Types**:

- `feat`: New feature (user-visible feature addition)
- `fix`: Bug fix

**Optional Types**:

- `build`: Build system or external dependency changes
- `chore`: Other changes (no release impact)
- `ci`: CI configuration files and scripts changes
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (whitespace, formatting, semicolons, etc.)
- `refactor`: Code changes without bug fixes or feature additions
- `perf`: Performance improvements
- `test`: Adding or fixing tests

### Output Example (English Project)

```bash
$ /commit-message

📝 Commit Message Suggestions
━━━━━━━━━━━━━━━━━━━━━━━━━

✨ Main Candidate:
feat: implement JWT-based authentication system

📋 Alternatives:
1. feat: add user authentication with JWT tokens
2. fix: resolve token validation error in auth middleware
3. refactor: extract auth logic into separate module

✅ `git commit -m "feat: implement JWT-based authentication system"` copied to clipboard
```

**Implementation Example (Fixed)**:

```bash
# Copy commit command to clipboard first (no newline)
printf 'git commit -m "%s"' "$COMMIT_MESSAGE" | pbcopy

# Then display message
cat << EOF
📝 Commit Message Suggestions
━━━━━━━━━━━━━━━━━━━━━━━━━

✨ Main Candidate:
$COMMIT_MESSAGE

📋 Alternatives:
1. ...
2. ...
3. ...

✅ \`git commit -m "$COMMIT_MESSAGE"\` copied to clipboard
EOF
```

### Output Example (Japanese Project)

```bash
$ /commit-message

📝 Commit Message Suggestions
━━━━━━━━━━━━━━━━━━━━━━━━━

✨ Main Candidate:
feat: JWT authentication system implemented

📋 Alternatives:
1. feat: add user authentication with JWT tokens
2. fix: resolve token validation error in auth middleware
3. docs: separate auth logic into different module

✅ `git commit -m "feat: JWT authentication system implemented"` copied to clipboard
```

### Operation Overview

1. **Analysis**: Analyze content of `git diff --staged`
2. **Generation**: Generate appropriate commit message
3. **Copy**: Automatically copy main candidate to clipboard

**Note**: This command does not execute git add or git commit. It only generates commit messages and copies to clipboard.

### Smart Features

#### 1. Automatic Change Classification (Staged Files Only)

- New file addition → `feat`
- Error fix patterns → `fix`
- Test files only → `test`
- Configuration file changes → `chore`
- README/docs updates → `docs`

#### 2. Automatic Project Convention Detection

- `.gitmessage` file
- Conventions in `CONTRIBUTING.md`
- Past commit history patterns

#### 3. Language Detection Details (Staged Changes Only)

```bash
# Detection criteria (priority order)
1. Detect language from git diff --staged content
2. Comment analysis of staged files
3. Language analysis of git log --oneline -20
4. Project main language settings
```

#### 4. Staging Analysis Details

Information used for analysis (read-only):

- `git diff --staged --name-only` - Changed file list
- `git diff --staged` - Actual change content
- `git status --porcelain` - File status

### Breaking Change Detection

For breaking API changes:

**English**:

```bash
feat!: change user API response format

BREAKING CHANGE: user response now includes additional metadata
```

Or:

```bash
feat(api)!: change authentication flow
```

**Japanese**:

```bash
feat!: change user API response format

BREAKING CHANGE: response now includes additional metadata
```

Or:

```bash
feat(api)!: change authentication flow
```

### Best Practices

1. **Match project**: Follow existing commit language
2. **Conciseness**: Clear within 50 characters
3. **Consistency**: Don't mix languages (stay consistent in English)
4. **OSS**: English recommended for open source
5. **Single line**: Always single-line commit message (supplement with PR for detailed explanations)

### Common Patterns

**English**:

```
feat: add user registration endpoint
fix: resolve memory leak in cache manager
docs: update API documentation
```

**Japanese**:

```
feat: add user registration endpoint
fix: resolve memory leak in cache manager
docs: update API documentation
```

### Integration with Claude

```bash
# Use with staged changes
git add -p  # Interactive staging
/commit-message
"Generate optimal commit message"

# Stage and analyze specific files
git add src/auth/*.js
/commit-message --lang en
"Generate message for authentication changes"

# Breaking Change detection and handling
git add -A
/commit-message --breaking
"Mark appropriately if there are breaking changes"
```

### Important Notes

- **Prerequisite**: Changes must be staged with `git add` beforehand
- **Limitation**: Unstaged changes are not analyzed
- **Recommendation**: Check existing project commit conventions first
