## Flutter Dependencies Update

Safely update dependencies in your Flutter project.

### Usage

```bash
# Check dependency status and request Claude's help
flutter pub deps --style=compact
"Please update the dependencies in pubspec.yaml to their latest versions"
```

### Basic Examples

```bash
# Check current dependencies
cat pubspec.yaml
"Analyze this Flutter project's dependencies and tell me which packages can be updated"

# Check before upgrading
flutter pub upgrade --dry-run
"Check if there are any breaking changes in this planned upgrade"
```

### Integration with Claude

```bash
# Comprehensive dependency update
cat pubspec.yaml
"Analyze Flutter dependencies and perform the following:
1. Research the latest version of each package
2. Check for breaking changes
3. Evaluate risk level (safe, caution, dangerous)
4. Suggest necessary code changes
5. Generate updated pubspec.yaml"

# Safe, gradual update
flutter pub outdated
"Update only packages that can be safely updated, avoiding major version upgrades"

# Impact analysis for specific package update
"Tell me the impact and necessary changes when updating provider to the latest version"
```

### Detailed Examples

```bash
# Detailed analysis including release notes
cat pubspec.yaml && flutter pub outdated
"Analyze dependencies and provide the following for each package in table format:
1. Current → Latest version
2. Risk evaluation (safe, caution, dangerous)
3. Main changes (from CHANGELOG)
4. Required code fixes"

# Null Safety migration analysis
cat pubspec.yaml
"Identify packages not compatible with Null Safety and create a migration plan"
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
cp pubspec.yaml pubspec.yaml.backup
cp pubspec.lock pubspec.lock.backup

# Execute update
flutter pub upgrade

# Verify after update
flutter analyze
flutter test
flutter pub deps --style=compact
```

### Notes

Always verify functionality after updates. If issues occur, restore with:

```bash
cp pubspec.yaml.backup pubspec.yaml
cp pubspec.lock.backup pubspec.lock
flutter pub get
```
