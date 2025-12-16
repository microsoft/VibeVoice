## Role Help

A selection guide and help system when you're unsure which role to use.

### Usage

```bash
/role-help                      # General role selection guide
/role-help <situation/problem>  # Recommended roles for specific situations
/role-help compare <Role 1>,<Role 2> # Compare roles
```

### Basic Examples

```bash
# General guidance
/role-help
→ List of available roles and their characteristics

# Situation-specific recommendation
/role-help "Concerned about API security"
→ Recommendation and usage of security role

# Role comparison
/role-help compare frontend,mobile
→ Differences and appropriate usage between frontend and mobile roles
```

### Situation-Based Role Selection Guide

### Security-Related

```
Use security role for:
✅ Implementation of login/authentication functions
✅ Security vulnerability checks for APIs
✅ Data encryption and privacy protection
✅ Security compliance verification
✅ Penetration testing

Usage: /role security
```

### 🏗️ Architecture & Design

```
Use architect role for:
✅ Evaluation of overall system design
✅ Microservices vs monolith decisions
✅ Database design and technology selection
✅ Scalability and extensibility considerations
✅ Technical debt assessment and improvement planning

Usage: /role architect
```

### ⚡ Performance Issues

```
Use performance role for:
✅ Slow applications
✅ Database query optimization
✅ Web page loading speed improvement
✅ Memory and CPU usage optimization
✅ Scaling and load countermeasures

Usage: /role performance
```

### 🔍 Problem Root Cause Investigation

```
Use analyzer role for:
✅ Root cause analysis of bugs and errors
✅ Investigation of system failures
✅ Structural analysis of complex problems
✅ Data analysis and statistical research
✅ Understanding why problems occur

Usage: /role analyzer
```

### 🎨 Frontend & UI/UX

```
Use frontend role for:
✅ User interface improvements
✅ Accessibility compliance
✅ Responsive design
✅ Usability and ease of use enhancement
✅ General web frontend technologies

Usage: /role frontend
```

### 📱 Mobile App Development

```
Use mobile role for:
✅ iOS and Android app optimization
✅ Mobile-specific UX design
✅ Touch interface optimization
✅ Offline support and synchronization functions
✅ App Store and Google Play compliance

Usage: /role mobile
```

### 👀 Code Review & Quality

```
Use reviewer role for:
✅ Code quality checks
✅ Readability and maintainability evaluation
✅ Coding convention verification
✅ Refactoring proposals
✅ PR and commit reviews

Usage: /role reviewer
```

### 🧪 Testing & Quality Assurance

```
Use qa role for:
✅ Test strategy planning
✅ Test coverage evaluation
✅ Automated test implementation guidelines
✅ Bug prevention and quality improvement measures
✅ Test automation in CI/CD

Usage: /role qa
```

### When Multiple Roles Are Needed

### 🔄 multi-role (Parallel Analysis)

```
Use multi-role for:
✅ Evaluation from multiple professional perspectives
✅ Creating integrated improvement plans
✅ Comparing evaluations from different fields
✅ Organizing contradictions and overlaps

Example: /multi-role security,performance
```

### 🗣️ role-debate (Discussion)

```
Use role-debate for:
✅ Trade-offs between specialized fields
✅ Divided opinions on technology selection
✅ Making design decisions through discussion
✅ Hearing debates from different perspectives

Example: /role-debate security,performance
```

### 🤖 smart-review (Automatic Proposal)

```
Use smart-review for:
✅ Uncertainty about which role to use
✅ Wanting to know the optimal approach for current situation
✅ Choosing from multiple options
✅ Beginner indecision

Example: /smart-review
```

### Role Comparison Table

### Security Category

| Role | Main Use | Strengths | Weaknesses |
|------|----------|----------|----------|
| security | Vulnerability and attack countermeasures | Threat analysis, authentication design | UX, performance |
| analyzer | Root cause analysis | Logical analysis, evidence collection | Preventive measures, future planning |

### Design Category

| Role | Main Use | Strengths | Weaknesses |
|------|----------|----------|----------|
| architect | System design | Long-term perspective, overall optimization | Detailed implementation, short-term solutions |
| reviewer | Code quality | Implementation level, maintainability | Business requirements, UX |

### Performance Category

| Role | Main Use | Strengths | Weaknesses |
|------|----------|----------|----------|
| performance | Speed improvement and optimization | Measurement, bottleneck identification | Security, UX |
| qa | Quality assurance | Testing, automation | Design, architecture |

### User Experience Category

| Role | Main Use | Strengths | Weaknesses |
|------|----------|----------|----------|
| frontend | Web UI/UX | Browser, accessibility | Server-side, DB |
| mobile | Mobile UX | Touch, offline support | Server-side, Web |

### Decision Flowchart When Unsure

```
What is the nature of the problem?
├─ Security-related → security
├─ Performance issues → performance
├─ Bug/failure investigation → analyzer
├─ UI/UX improvement → frontend or mobile
├─ Design/architecture → architect
├─ Code quality → reviewer
├─ Testing-related → qa
└─ Complex/composite → smart-review for proposal

Spans multiple fields?
├─ Want integrated analysis → multi-role
├─ Discussion/trade-offs → role-debate
└─ Unsure → smart-review
```

### Frequently Asked Questions

### Q: What's the difference between frontend and mobile roles?

```
A:
frontend: Web browser-focused, HTML/CSS/JavaScript
mobile: Mobile app-focused, iOS/Android native, React Native, etc.

For issues related to both, multi-role frontend,mobile is recommended
```

### Q: How to choose between security and analyzer roles?

```
A:
security: Prevention of attacks and threats, security design
analyzer: Analysis of causes of existing problems, investigation

For security incident investigations, use multi-role security,analyzer
```

### Q: What's the difference between architect and performance roles?

```
A:
architect: Long-term design of entire systems, scalability
performance: Specific speed and efficiency improvements

For performance design of large-scale systems, use multi-role architect,performance
```

### Collaboration with Claude

```bash
# Combined with situation description
/role-help
"React app page loading is slow, receiving complaints from users"

# Combined with file content
cat problem-description.md
/role-help
"Recommend the most suitable role for this problem"

# When unsure between specific options
/role-help compare security,performance
"Which role is appropriate for JWT token expiration issues?"
```

### Notes

- For complex problems, combining multiple roles is more effective
- For urgent matters, use single role for quick response
- When unsure, it's recommended to use smart-review for automatic proposals
- The final decision should be made by the user considering the nature of the problem
