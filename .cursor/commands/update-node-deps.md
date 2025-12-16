## Node Dependencies Update

Safely update dependencies in your Node.js project.

### Usage

```bash
# Check dependency status and request Claude's help
npm outdated
"Please update the dependencies in package.json to their latest versions"
```

### Basic Examples

```bash
# Check current dependencies
cat package.json
"Analyze this Node.js project's dependencies and tell me which packages can be updated"

# Check list of updatable packages
npm outdated
"Analyze the risk level of updating these packages"
```

### Integration with Claude

```bash
# Comprehensive dependency update
cat package.json
"Analyze Node.js dependencies and perform the following:
1. Research the latest version of each package
2. Check for breaking changes
3. Evaluate risk level (safe, caution, dangerous)
4. Suggest necessary code changes
5. Generate updated package.json"

# Safe, gradual update
npm outdated
"Update only packages that can be safely updated, avoiding major version upgrades"

# Impact analysis for specific package update
"Tell me the impact and necessary changes when updating express to the latest version"
```

### Detailed Examples

```bash
# Detailed analysis including release notes
cat package.json && npm outdated
"Analyze dependencies and provide the following for each package in table format:
1. Current → Latest version
2. Risk evaluation (safe, caution, dangerous)
3. Main changes (from CHANGELOG)
4. Required code fixes"

# TypeScript project with type definitions consideration
cat package.json tsconfig.json
"Update dependencies including TypeScript type definitions and create an update plan that avoids type errors"
```

### Risk Criteria

```
Safe (🟢):
- Patch version upgrade (1.2.3 → 1.2.4)
- Bug fixes only
- Backward compatibility guaranteed

Caution (🟡):
- Minor version upgrade (1.2.3 → 1.3.0)
- New features added
- Deprecation warnings

Dangerous (🔴):
- Major version upgrade (1.2.3 → 2.0.0)
- Breaking changes
- API removals or modifications
```

### Execution of Update

```bash
# Create backups
cp package.json package.json.backup
cp package-lock.json package-lock.json.backup

# Execute update
npm update

# Verify after update
npm test
npm run build
npm audit
```

### Notes

Always verify functionality after updates. If issues occur, restore with:

```bash
cp package.json.backup package.json
cp package-lock.json.backup package-lock.json
npm install
```
