## Role

Switch to a specific role to perform specialized analysis or work.

### Usage

```bash
/role <role_name> [--agent|-a]
```

### Options

- `--agent` or `-a`: Execute as a sub-agent (recommended for large-scale analysis)
  - When using this option, if the role description includes proactive delegation phrases (such as "use PROACTIVELY"), more proactive automatic delegation will be enabled

### Available Roles

#### Specialized Analysis Roles (Evidence-First Integrated)

- `security`: Security audit expert (OWASP Top 10, threat modeling, Zero Trust principles, CVE matching)
- `performance`: Performance optimization expert (Core Web Vitals, RAIL model, phased optimization, ROI analysis)
- `analyzer`: Root cause analysis expert (5 Whys, systems thinking, hypothesis-driven, cognitive bias countermeasures)
- `frontend`: Frontend and UI/UX expert (WCAG 2.1, design systems, user-centered design)

#### Development Support Roles

- `reviewer`: Code review expert (readability, maintainability, performance, refactoring proposals)
- `architect`: System architect (Evidence-First design, MECE analysis, evolutionary architecture)
- `qa`: Test engineer (test coverage, E2E/integration/unit strategy, automation proposals)
- `mobile`: Mobile development expert (iOS HIG, Android Material Design, cross-platform strategy)

### Basic Examples

```bash
# Switch to security audit mode (normal)
/role security
"Check the security vulnerabilities of this project"

# Run security audit as a sub-agent (large-scale analysis)
/role security --agent
"Perform a security audit of the entire project"

# Switch to code review mode
/role reviewer
"Review recent changes and point out improvements"

# Switch to performance optimization mode
/role performance
"Analyze the bottlenecks of the application"

# Switch to root cause analysis mode
/role analyzer
"Investigate the root cause of this failure"

# Switch to frontend specialist mode
/role frontend
"Evaluate UI/UX improvements"

# Switch to mobile development specialist mode
/role mobile
"Evaluate mobile optimization of this app"

# Return to normal mode
/role default
"Return to normal Claude"
```

### Collaboration with Claude

```bash
# Security-specific analysis
/role security
cat app.js
"Analyze potential security risks in this code in detail"

# Architecture evaluation
/role architect
ls -la src/
"Present problems and improvements for the current structure"

# Test strategy planning
/role qa
"Propose the optimal test strategy for this project"
```

### Detailed Examples

```bash
# Analysis with multiple roles
/role security
"First check from a security perspective"
git diff HEAD~1

/role reviewer
"Next review general code quality"

/role architect
"Finally evaluate from an architectural perspective"

# Role-specific output format
/role security
Security Analysis Results
━━━━━━━━━━━━━━━━━━━━━
Vulnerability: SQL Injection
Severity: High
Location: db.js:42
Fix: Use parameterized queries
```

### Evidence-First Integration Features

#### Core Philosophy

Each role adopts an **Evidence-First** approach, conducting analysis and making proposals based not on speculation but on **proven methods, official guidelines, and objective data**.

#### Common Features

- **Official Documentation Compliance**: Prioritized reference to authoritative official guidelines in each field
- **MECE Analysis**: Systematic problem decomposition without omissions or duplicates
- **Multidimensional Evaluation**: Multiple perspectives (technical, business, operational, user)
- **Cognitive Bias Countermeasures**: Mechanisms to eliminate confirmation bias, etc.
- **Discussion Characteristics**: Role-specific professional discussion stances

### Details of Specialized Analysis Roles

#### security (Security Audit Expert)

**Evidence-Based Security Audit**

- Systematic evaluation according to OWASP Top 10, Testing Guide, and SAMM
- Known vulnerability checks through CVE and NVD database matching
- Threat modeling using STRIDE, Attack Tree, and PASTA
- Design evaluation based on Zero Trust principles and least privilege

**Professional Report Format**

```
Evidence-Based Security Audit Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OWASP Top 10 Compliance: XX% / CVE Matching: Completed
Threat Modeling: STRIDE Analysis Completed
```

#### performance (Performance Optimization Expert)

**Evidence-First Performance Optimization**

- Compliance with Core Web Vitals (LCP, FID, CLS) and RAIL model
- Implementation of Google PageSpeed Insights recommendations
- Phased optimization process (measurement → analysis → prioritization → implementation)
- Quantitative evaluation of ROI through analysis

**Professional Report Format**

```
Evidence-First Performance Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core Web Vitals: LCP[XXXms] FID[XXXms] CLS[X.XX]
Performance Budget: XX% / ROI Analysis: XX% Improvement Prediction
```

#### analyzer (Root Cause Analysis Expert)

**Evidence-First Root Cause Analysis**

- 5 Whys + α method (including counter-evidence examination)
- Structural analysis through systems thinking (Peter Senge principles)
- Cognitive bias countermeasures (elimination of confirmation bias, anchoring, etc.)
- Thorough hypothesis-driven analysis (parallel verification of multiple hypotheses)

**Professional Report Format**

```
Evidence-First Root Cause Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analysis Confidence: High / Bias Countermeasures: Implemented
Hypothesis Verification Matrix: XX% Confidence
```

#### frontend (Frontend & UI/UX Expert)

**Evidence-First Frontend Development**

- WCAG 2.1 accessibility compliance
- Material Design and iOS HIG official guidelines compliance
- User-centered design and design system standard application
- Verification through A/B testing and user behavior analysis

### Details of Development Support Roles

#### reviewer (Code Review Expert)

- Multidimensional evaluation of readability, maintainability, and performance
- Coding convention compliance checks and refactoring proposals
- Cross-cutting confirmation of security and accessibility

#### architect (System Architect)

- Evidence-First design principles and MECE analysis for phased thinking
- Evolutionary architecture and multi-perspective evaluation (technical, business, operational, user)
- Reference to official architecture patterns and best practices

#### qa (Test Engineer)

- Test coverage analysis and E2E/integration/unit test strategies
- Test automation proposals and quality metrics design

#### mobile (Mobile Development Expert)

- iOS HIG and Android Material Design official guidelines compliance
- Cross-platform strategy and Touch-First design
- Store review guidelines and mobile-specific UX optimization

### Role-Specific Discussion Characteristics

Each role has unique discussion stances, evidence sources, and strengths according to their specialized field.

#### security Role Discussion Characteristics

- **Stance**: Conservative approach, risk minimization priority, worst-case scenario assumption
- **Evidence**: OWASP guidelines, NIST frameworks, actual attack cases
- **Strengths**: Precision in risk assessment, deep knowledge of regulatory requirements, comprehensive understanding of attack methods
- **Caution**: Excessive conservatism, insufficient UX consideration, downplaying implementation costs

#### performance Role Discussion Characteristics

- **Stance**: Data-driven decisions, efficiency focus, user experience priority, continuous improvement
- **Evidence**: Core Web Vitals, benchmark results, user behavior data, industry standards
- **Strengths**: Quantitative evaluation ability, precision in bottleneck identification, ROI analysis
- **Caution**: Downplaying security, insufficient maintainability consideration, overemphasis on measurement

#### analyzer Role Discussion Characteristics

- **Stance**: Evidence-focused, hypothesis verification, structural thinking, bias elimination
- **Evidence**: Measured data, statistical methods, systems thinking theory, cognitive bias research
- **Strengths**: Logical analysis ability, objectivity in evidence evaluation, ability to discover structural problems
- **Caution**: Analysis paralysis, perfectionism, data absolutism, excessive skepticism

#### frontend Role Discussion Characteristics

- **Stance**: User-centered, accessibility-focused, design principle compliance, experience value priority
- **Evidence**: UX research, accessibility standards, design systems, usability testing
- **Strengths**: User perspective, design principles, accessibility, experience design
- **Caution**: Downplaying technical constraints, insufficient performance consideration, implementation complexity

### Effects of Multi-Role Collaboration

Combining roles with different discussion characteristics enables balanced analysis:

#### Typical Collaboration Patterns

- **security + frontend**: Balance between security and usability
- **performance + security**: Balance between speed and safety
- **analyzer + architect**: Integration of problem analysis and structural design
- **reviewer + qa**: Coordination of code quality and test strategy

## Advanced Role Features

### Intelligent Role Selection

- `/smart-review`: Automatic role proposal through project analysis
- `/role-help`: Optimal role selection guide according to the situation

### Multi-Role Collaboration

- `/multi-role <Role 1>,<Role 2>`: Simultaneous analysis by multiple roles
- `/role-debate <Role 1>,<Role 2>`: Discussion between roles

### Usage Examples

#### Automatic Role Proposal

```bash
/smart-review
→ Analyze current situation and propose optimal role

/smart-review src/auth/
→ Recommend security role based on authentication-related files
```

#### Multiple Role Analysis

```bash
/multi-role security,performance
"Evaluate this API from multiple perspectives"
→ Integrated analysis from both security and performance perspectives

/role-debate frontend,security
"Discuss the UX of 2-factor authentication"
→ Discussion from usability and security perspectives
```

#### When Unsure About Role Selection

```bash
/role-help "API is slow and security is also a concern"
→ Propose appropriate approach (multi-role or debate)

/role-help compare frontend,mobile
→ Differences and appropriate usage between frontend and mobile roles
```

## Notes

### About Role Behavior

- When switching roles, Claude's **behavior, priorities, analysis methods, and report formats** become specialized
- Each role prioritizes applying official guidelines and proven methods through an **Evidence-First approach**
- Return to normal mode with `default` (role specialization is removed)
- Roles are only effective within the current session

### Effective Usage Methods

- **Simple problems**: Sufficient specialized analysis with a single role
- **Complex problems**: Multi-perspective analysis with multi-role or role-debate is effective
- **When unsure**: Use smart-review or role-help
- **Continuous improvement**: Even with the same role, analysis accuracy improves with new evidence and methods

### Sub-Agent Function (--agent Option)

For large-scale analysis or independent specialized processing, you can run a role as a sub-agent using the `--agent` option.

#### Benefits

- **Independent context**: Does not interfere with main conversation
- **Parallel execution**: Multiple analyses can be performed simultaneously
- **Specialized expertise**: Deeper analysis and detailed reports
- **Promotion of automatic delegation**: When role descriptions include "use PROACTIVELY" or "MUST BE USED", more proactive automatic delegation is enabled

#### Recommended Usage Scenarios

```bash
# Security: OWASP full-item check, CVE matching
/role security --agent
"Security audit of entire codebase"

# Analyst: Root cause analysis of large logs
/role analyzer --agent
"Analyze error logs from the past week"

# Reviewer: Detailed review of large PR
/role reviewer --agent
"Review 1000-line changes in PR #500"
```

#### Normal Role vs Sub-Agent

| Situation | Recommendation | Command |
|-----------|----------------|---------|
| Simple confirmation | Normal role | `/role security` |
| Large-scale analysis | Sub-agent | `/role security --agent` |
| Interactive work | Normal role | `/role frontend` |
| Independent audit | Sub-agent | `/role qa --agent` |

### Role Configuration Details

- Detailed settings, expertise, and discussion characteristics of each role are defined in the `.claude/agents/roles/` directory
- Includes Evidence-First methods and cognitive bias countermeasures
- Specialized mode is automatically enabled with role-specific trigger phrases
- Actual role files consist of over 200 lines of professional content
