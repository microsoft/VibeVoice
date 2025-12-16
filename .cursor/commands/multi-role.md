## Multi Role

A command that analyzes the same target in parallel with multiple roles and generates an integrated report.

### Usage

```bash
/multi-role <role1>,<role2> [--agent|-a] [analysis_target]
/multi-role <role1>,<role2>,<role3> [--agent|-a] [analysis_target]
```

**Important**:

- Place the `--agent` option immediately after specifying roles
- Write your message after `--agent`
- Correct example: `/multi-role qa,architect --agent Evaluate the plan`
- Incorrect example: `/multi-role qa,architect Evaluate the plan --agent`

### Options

- `--agent` or `-a`: Execute each role as a sub-agent in parallel (recommended for large-scale analysis)
  - When using this option, if role descriptions include proactive delegation phrases (like "use PROACTIVELY"), more aggressive automatic delegation becomes enabled

### Basic Examples

```bash
# Dual analysis of security and performance (normal)
/multi-role security,performance
"Evaluate this API endpoint"

# Parallel analysis of large-scale system (sub-agents)
/multi-role security,performance --agent
"Comprehensively analyze system security and performance"

# Integrated analysis of frontend, mobile, and performance
/multi-role frontend,mobile,performance
"Consider optimization proposals for this screen"

# Multifaceted evaluation of architecture design (sub-agents)
/multi-role architect,security,performance --agent
"Evaluate microservices design"
```

### Analysis Process

### Phase 1: Parallel Analysis

Each role independently analyzes the same target

- Perform evaluation from specialized perspective
- Make judgments based on role-specific criteria
- Generate independent recommendations

### Phase 2: Integrated Analysis

Structure and integrate results

- Organize evaluation results from each role
- Identify overlaps and contradictions
- Clarify complementary relationships

### Phase 3: Integrated Report

Generate final recommendations

- Prioritized action plan
- Explicit trade-offs
- Implementation roadmap

### Output Format Examples

### For 2-role Analysis

```
Multi-role Analysis: Security + Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analysis Target: API endpoint /api/users

Security Analysis Results:
Authentication: JWT verification properly implemented
Authorization: Role-based access control incomplete
Encryption: API keys logged in plain text

Evaluation Score: 65/100
Importance: High (due to sensitive data access)

Performance Analysis Results:
Response Time: Average 180ms (within target of 200ms)
Database Queries: N+1 problem detected
Caching: Redis cache not implemented

Evaluation Score: 70/100
Importance: Medium (currently within acceptable range)

Interrelated Analysis:
Synergistic Opportunities:
- Consider encryption when implementing Redis cache
- Improve logging for both security and performance gains

Trade-off Points:
- Authorization check strengthening ↔ Impact on response time
- Log encryption ↔ Reduced debugging efficiency

Integrated Priorities:
Critical: Fix API key plain text output
High: Resolve N+1 queries
Medium: Implement Redis cache + encryption
Low: Refine authorization control

Implementation Roadmap:
Week 1: Implement API key masking
Week 2: Database query optimization
Weeks 3-4: Cache layer design and implementation
Month 2: Progressive strengthening of authorization control
```

### For 3-role Analysis

```
Multi-role Analysis: Frontend + Mobile + Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analysis Target: User Profile Screen

Frontend Analysis Results:
Usability: Intuitive layout
Accessibility: 85% WCAG 2.1 compliance
Responsive: Issues with tablet display

Mobile Analysis Results:
Touch Targets: 44pt+ ensured
One-handed Operation: Important buttons placed at top
Offline Support: Not implemented

Performance Analysis Results:
Initial Display: LCP 2.1s (good)
Image Optimization: WebP not supported
Lazy Loading: Not implemented

Integrated Recommendations:
1. Mobile optimization (one-handed operation + offline support)
2. Image optimization (WebP + lazy loading)
3. Tablet UI improvements

Priority: Mobile > Performance > Frontend
Implementation Period: 3-4 weeks
```

### Effective Combination Patterns

### Security-focused

```bash
/multi-role security,architect
"Authentication system design"

/multi-role security,frontend
"Login screen security"

/multi-role security,mobile
"Mobile app data protection"
```

### Performance-focused

```bash
/multi-role performance,architect
"Scalability design"

/multi-role performance,frontend
"Web page speed optimization"

/multi-role performance,mobile
"App performance optimization"
```

### User Experience-focused

```bash
/multi-role frontend,mobile
"Cross-platform UI"

/multi-role frontend,performance
"Balance between UX and performance"

/multi-role mobile,performance
"Mobile UX optimization"
```

### Comprehensive Analysis

```bash
/multi-role architect,security,performance
"Overall system evaluation"

/multi-role frontend,mobile,performance
"Comprehensive user experience evaluation"

/multi-role security,performance,mobile
"Comprehensive mobile app diagnosis"
```

### Collaboration with Claude

```bash
# Combine with file analysis
cat src/components/UserProfile.tsx
/multi-role frontend,mobile
"Evaluate this component from multiple perspectives"

# Evaluate design documents
cat architecture-design.md
/multi-role architect,security,performance
"Evaluate this design across multiple specialties"

# Error analysis
cat performance-issues.log
/multi-role performance,analyzer
"Analyze performance issues from multiple angles"
```

### Choosing between multi-role and role-debate

### When to use multi-role

- You want independent evaluations from each specialty
- You want to create an integrated improvement plan
- You want to organize contradictions and overlaps
- You want to determine implementation priorities

### When to use role-debate

- There are trade-offs between specialties
- Opinions might differ on technology selection
- You want to decide design policies through discussion
- You want to hear debates from different perspectives

### Sub-agent Parallel Execution (--agent)

Using the `--agent` option executes each role as an independent sub-agent in parallel.

#### Promoting Automatic Delegation

If role file descriptions include phrases like these, more proactive automatic delegation is enabled when using `--agent`:

- "use PROACTIVELY"
- "MUST BE USED"
- Other emphasis expressions

#### Execution Flow

```
Normal execution:
Role 1 → Role 2 → Role 3 → Integration
(Sequential execution, approx. 3-5 minutes)

--agent execution:
Role 1 ─┐
Role 2 ─┼→ Integration
Role 3 ─┘
(Parallel execution, approx. 1-2 minutes)
```

#### Effective Usage Examples

```bash
# Comprehensive evaluation of large-scale system
/multi-role architect,security,performance,qa --agent
"Comprehensive evaluation of new system"

# Detailed analysis from multiple perspectives
/multi-role frontend,mobile,performance --agent
"Full screen UX optimization analysis"
```

#### Performance Comparison

| Number of Roles | Normal Execution | --agent Execution | Reduction Rate |
|----------------|-----------------|------------------|---------------|
| 2 roles        | 2-3 minutes     | 1 minute         | 50%           |
| 3 roles        | 3-5 minutes     | 1-2 minutes      | 60%           |
| 4 roles        | 5-8 minutes     | 2-3 minutes      | 65%           |

### Notes

- Executing 3 or more roles simultaneously results in longer output
- Complex analyses may take longer to execute
- If conflicting recommendations arise, consider using role-debate
- Final judgments should be made by the user with reference to integrated results
- **When using --agent**: Consumes more resources but is efficient for large-scale analyses
