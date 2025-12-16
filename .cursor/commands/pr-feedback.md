## PR Feedback

Efficiently handle Pull Request review comments and achieve root cause resolution using a 3-stage error analysis approach.

### Usage

```bash
# Retrieve and analyze review comments
gh pr view --comments
"Classify review comments by priority and create an action plan"

# Detailed analysis of error-related comments
gh pr checks
"Analyze CI errors using a 3-stage approach to identify root causes"

# Quality confirmation after fixes
npm test && npm run lint
"Fixes are complete - please check regression tests and code quality"
```

### Basic Examples

```bash
# Classify comments
gh pr view 123 --comments | head -20
"Classify review comments into must/imo/nits/q categories and determine response order"

# Collect error information
npm run build 2>&1 | tee error.log
"Identify the root cause of build errors and suggest appropriate fixes"

# Verify fix implementation
git diff HEAD~1
"Evaluate whether this fix appropriately addresses the review comments"
```

### Comment Classification System

```
🔴 must: Required fixes
├─ Security issues
├─ Functional bugs
├─ Design principle violations
└─ Convention violations

🟡 imo: Improvement suggestions
├─ Better implementation methods
├─ Performance improvements
├─ Readability enhancements
└─ Refactoring proposals

🟢 nits: Minor points
├─ Typo fixes
├─ Indentation adjustments
├─ Comment additions
└─ Naming refinements

🔵 q: Questions/confirmations
├─ Implementation intent verification
├─ Specification clarification
├─ Design decision background
└─ Alternative solution consideration
```

### 3-Stage Error Analysis Approach

#### Stage 1: Information Collection

**Required actions**

- Full error message capture
- Stack trace review
- Reproduction condition identification

**Recommended actions**

- Environment information collection
- Recent change history
- Related logs review

#### Stage 2: Root Cause Analysis

- 5 Whys analysis application
- Dependency tracking
- Environment difference checking
- Minimal reproduction code creation

#### Stage 3: Solution Implementation

- Immediate response (hotfix)
- Root cause resolution (essential fix)
- Preventive measures (recurrence prevention)

### Response Flow

1. **Comment analysis**: Classification by priority
2. **Fix plan**: Determining response order
3. **Phased fixes**: Critical → High → Medium → Low
4. **Quality confirmation**: Testing, linting, building
5. **Progress report**: Description of specific fixes

### Post-Fix Verification

```bash
# Basic checks
npm test
npm run lint
npm run build

# Regression tests
npm run test:e2e

# Code quality
npm run test:coverage
```

### Reply Templates

**Fix completion report**

```markdown
@reviewer Thank you for your feedback.
Fixes are complete:

- [Specific fix details]
- [Test results]
- [Verification method]
```

**Technical decision explanation**

```markdown
Implementation background: [Reason]
Considered alternatives: [Options and decision rationale]
Adopted solution benefits: [Advantages]
```

### Notes

- **Priority adherence**: Address in order of Critical → High → Medium → Low
- **Test first**: Confirm regression tests before making fixes
- **Clear reporting**: Describe fix details and verification methods specifically
- **Constructive dialogue**: Polite communication based on technical grounds
