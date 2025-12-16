## Analyze Performance

Finds performance bottlenecks and suggests fixes based on technical debt analysis.

### Usage

```bash
# Find performance issues comprehensively
find . -name "*.js" -o -name "*.ts" | xargs wc -l | sort -rn | head -10
"Show me the big files and performance problems, then suggest fixes"

# Spot inefficient patterns
grep -r "for.*await\|forEach.*await" . --include="*.js"
"Find N+1 queries and other performance killers"

# Check for memory leaks
grep -r "addEventListener\|setInterval" . --include="*.js" | grep -v "removeEventListener\|clearInterval"
"Where might we have memory leaks and how do we fix them?"
```

### Basic Examples

```bash
# Check bundle size and load time
npm ls --depth=0 && find ./public -name "*.js" -o -name "*.css" | xargs ls -lh
"How can we shrink bundles and optimize assets?"

# Database query performance
grep -r "SELECT\|findAll\|query" . --include="*.js" | head -20
"Which database queries need optimization?"

# Dependency performance impact
npm outdated && npm audit
"Are outdated dependencies slowing us down?"
```

### What We Look For

#### 1. Code-Level Problems

- **O(n²) Algorithms**: Slow array operations that don't scale
- **Synchronous I/O**: Blocking operations that freeze everything
- **Redundant Processing**: Doing the same work over and over
- **Memory Leaks**: Event listeners and timers left running

#### 2. Architecture Problems

- **N+1 Queries**: Too many database round trips
- **Missing Cache**: Repeating expensive operations
- **Bundle Bloat**: Shipping code users don't need
- **Resource Waste**: Poor connection and thread management

#### 3. Technical Debt Impact

- **Legacy Code**: Old implementations dragging us down
- **Poor Design**: Everything's too tightly coupled
- **Missing Tests**: Performance regressions slip through
- **Blind Spots**: Can't see problems until it's too late

### Improvement Priorities

```
🔴 Critical: System failure risk
├─ Memory leaks (server crashes)
├─ N+1 queries (database load)
└─ Synchronous I/O (response delays)

🟡 High: User experience impact
├─ Bundle size (initial load time)
├─ Image optimization (display speed)
└─ Caching strategy (response speed)

🟢 Medium: Operational efficiency
├─ Dependency updates (security)
├─ Code duplication (maintainability)
└─ Enhanced monitoring (operational load)
```

### Measurement and Tools

#### Node.js / JavaScript

```bash
# Profiling
node --prof app.js
clinic doctor -- node app.js

# Bundle analysis
npx webpack-bundle-analyzer
lighthouse --chrome-flags="--headless"
```

#### Database

```sql
-- Query analysis
EXPLAIN ANALYZE SELECT ...
SHOW SLOW LOG;
```

#### Frontend

```bash
# React performance
grep -r "useMemo\|useCallback" . --include="*.jsx"

# Resource analysis
find ./src -name "*.png" -o -name "*.jpg" | xargs ls -lh
```

### Keep Improving

- **Weekly checks**: Run performance tests regularly
- **Track metrics**: Watch response times and memory usage
- **Set alerts**: Get notified when things slow down
- **Share knowledge**: Document what works and what doesn't
