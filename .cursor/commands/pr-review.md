## PR Review

Ensure code quality and architectural soundness through systematic Pull Request reviews.

### Usage

```bash
# Comprehensive PR review
gh pr view 123 --comments
"Systematically review this PR and provide feedback from code quality, security, and architecture perspectives"

# Security-focused review
gh pr diff 123
"Focus on reviewing security risks and vulnerabilities"

# Architecture perspective review
gh pr checkout 123 && find . -name "*.js" | head -10
"Evaluate the architecture from the perspectives of layer separation, dependencies, and SOLID principles"
```

### Basic Examples

```bash
# Quantitative code quality assessment
find . -name "*.js" -exec wc -l {} + | sort -rn | head -5
"Evaluate code complexity, function size, and duplication, and point out improvements"

# Security vulnerability check
grep -r "password\|secret\|token" . --include="*.js" | head -10
"Check for risks of sensitive information leakage, hardcoding, and authentication bypass"

# Architecture violation detection
grep -r "import.*from.*\.\./\.\." . --include="*.js"
"Evaluate layer violations, circular dependencies, and coupling issues"
```

### Comment Classification System

```
🔴 critical.must: Critical issues
├─ Security vulnerabilities
├─ Data integrity problems
└─ System failure risks

🟡 high.imo: High-priority improvements
├─ Risk of malfunction
├─ Performance issues
└─ Significant decrease in maintainability

🟢 medium.imo: Medium-priority improvements
├─ Readability enhancement
├─ Code structure improvement
└─ Test quality improvement

🟢 low.nits: Minor points
├─ Style unification
├─ Typo fixes
└─ Comment additions

🔵 info.q: Questions/information
├─ Implementation intent confirmation
├─ Design decision background
└─ Best practices sharing
```

### Review Perspectives

#### 1. Code Correctness

- **Logic errors**: Boundary values, null checks, exception handling
- **Data integrity**: Type safety, validation
- **Error handling**: Completeness, appropriate processing

#### 2. Security

- **Authentication/authorization**: Appropriate checks, permission management
- **Input validation**: SQL injection, XSS countermeasures
- **Sensitive information**: Logging restrictions, encryption

#### 3. Performance

- **Algorithms**: Time complexity, memory efficiency
- **Database**: N+1 queries, index optimization
- **Resources**: Memory leaks, cache utilization

#### 4. Architecture

- **Layer separation**: Dependency direction, appropriate separation
- **Coupling**: Tight coupling, interface utilization
- **SOLID principles**: Single responsibility, open-closed, dependency inversion

### Review Flow

1. **Pre-check**: PR information, change diff, related issues
2. **Systematic checks**: Security → Correctness → Performance → Architecture
3. **Constructive feedback**: Specific improvement suggestions and code examples
4. **Follow-up**: Fix confirmation, CI status, final approval

### Effective Comment Examples

**Security Issues**

```markdown
**critical.must.** Password is stored in plaintext

```javascript
// Proposed fix
const bcrypt = require('bcrypt');
const hashedPassword = await bcrypt.hash(password, 12);
```

Hashing is required to prevent security risks.

```

**Performance Improvement**
```markdown
**high.imo.** N+1 query problem occurs

```javascript
// Improvement: Eager Loading
const users = await User.findAll({ include: [Post] });
```

This can significantly reduce the number of queries.

```

**Architecture Violation**
```markdown
**high.must.** Layer violation occurred

The domain layer directly depends on the infrastructure layer.
Please introduce an interface following the dependency inversion principle.
```

### Notes

- **Constructive tone**: Collaborative rather than aggressive communication
- **Specific suggestions**: Provide solutions along with pointing out problems
- **Prioritization**: Address in order of Critical → High → Medium → Low
- **Continuous improvement**: Document review results in a knowledge base
