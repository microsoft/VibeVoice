## Rust Dependencies Update

Safely update dependencies in your Rust project.

### Usage

```bash
# Check dependency status and request Claude's help
cargo tree
"Please update the dependencies in Cargo.toml to their latest versions"
```

### Basic Examples

```bash
# Check current dependencies
cat Cargo.toml
"Analyze this Rust project's dependencies and tell me which crates can be updated"

# Check list of updatable crates
cargo update --dry-run
"Analyze the risk level of updating these crates"
```

### Integration with Claude

```bash
# Comprehensive dependency update
cat Cargo.toml
"Analyze Rust dependencies and perform the following:
1. Research the latest version of each crate
2. Check for breaking changes
3. Evaluate risk level (safe, caution, dangerous)
4. Suggest necessary code changes
5. Generate updated Cargo.toml"

# Safe, gradual update
cargo tree
"Update only crates that can be safely updated, avoiding major version upgrades"

# Impact analysis for specific crate update
"Tell me the impact and necessary changes when updating tokio to the latest version"
```

### Detailed Examples

```bash
# Detailed analysis including release notes
cat Cargo.toml && cargo tree
"Analyze dependencies and provide the following for each crate in table format:
1. Current → Latest version
2. Risk evaluation (safe, caution, dangerous)
3. Main changes (from CHANGELOG)
4. Trait bound changes
5. Required code fixes"

# Async runtime migration analysis
cat Cargo.toml src/main.rs
"Present all necessary changes for migrating from async-std to tokio or upgrading tokio to a new major version"
```

### Risk Criteria

```
Safe (🟢):
- Patch version upgrade (0.1.2 → 0.1.3)
- Bug fixes only
- Backward compatibility guaranteed

Caution (🟡):
- Minor version upgrade (0.1.0 → 0.2.0)
- New features added
- Deprecation warnings

Dangerous (🔴):
- Major version upgrade (0.x.y → 1.0.0, 1.x.y → 2.0.0)
- Breaking changes
- API removals or modifications
- Trait bound changes
```

### Execution of Update

```bash
# Create backups
cp Cargo.toml Cargo.toml.backup
cp Cargo.lock Cargo.lock.backup

# Execute update
cargo update

# Verify after update
cargo check
cargo test
cargo clippy
```

### Notes

Always verify functionality after updates. If issues occur, restore with:

```bash
cp Cargo.toml.backup Cargo.toml
cp Cargo.lock.backup Cargo.lock
cargo build
```
