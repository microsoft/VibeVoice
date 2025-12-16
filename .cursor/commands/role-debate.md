## Role Debate

A command that allows roles with different expertise to discuss and examine trade-offs to derive optimal solutions.

### Usage

```bash
/role-debate <Role 1>,<Role 2> [Topic]
/role-debate <Role 1>,<Role 2>,<Role 3> [Topic]
```

### Basic Examples

```bash
# Security vs Performance trade-off
/role-debate security,performance
"JWT Token Expiry Setting"

# Usability vs Security balance
/role-debate frontend,security
"2-Factor Authentication UX Optimization"

# Technology selection discussion
/role-debate architect,mobile
"React Native vs Flutter Selection"

# Three-party debate
/role-debate architect,security,performance
"Pros and Cons of Microservices"
```

### Basic Principles of Debate

#### Constructive Debate Guidelines

- **Mutual Respect**: Respect the expertise and perspectives of other roles
- **Fact-Based**: Debate based on data and evidence, not emotional reactions
- **Solution-Oriented**: Aim for better solutions rather than criticizing for criticism's sake
- **Implementation-Focused**: Consider feasibility rather than idealism

#### Quality Requirements for Arguments

- **Official Documentation**: Reference standards, guidelines, and official documentation
- **Empirical Cases**: Specific citations of success or failure cases
- **Quantitative Evaluation**: Comparisons using numbers and metrics whenever possible
- **Time-Series Consideration**: Evaluation of short-term, medium-term, and long-term impacts

#### Debate Ethics

- **Honesty**: Acknowledge the limits of your expertise
- **Openness**: Flexibility toward new information and perspectives
- **Transparency**: Explicitly state judgment grounds and assumptions
- **Accountability**: Mention implementation risks of proposals

### Debate Process

### Phase 1: Initial Position Statement

Each role independently expresses opinions from their professional perspective

- Presentation of recommendations
- Explicit citation of standards and documents as grounds
- Explanation of anticipated risks and issues
- Definition of success metrics

### Phase 2: Mutual Discussion & Rebuttal

Cross-discussion between roles

- Constructive rebuttal of other roles' proposals
- Identification of overlooked perspectives
- Clarification of trade-offs
- Presentation of alternatives

### Phase 3: Search for Compromises

Exploration of implementable solutions

- Evaluation of the importance of each perspective
- Consideration of win-win solutions
- Step-by-step implementation approach
- Consideration of risk mitigation measures

### Phase 4: Integrated Conclusion

Determining final recommendations

- Agreed-upon solution
- Implementation roadmap
- Success metrics and measurement methods
- Future review points

### Output Format Examples

### For 2-Role Debate

```
Role Debate: Security vs Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Topic: JWT Token Expiry Setting

Security Role's Argument:
"15-minute short expiry recommended"

Grounds:
- Compliance with OWASP JWT Security Cheat Sheet
- Minimizing damage window in case of token leakage
- Limiting attacker available time

Concerns:
- Long expiry exponentially increases attack risk
- Short expiry mandatory for financial compliance

Success Metrics:
- Security incident rate < 0.1%
- Average attack detection time < 5 minutes

Performance Role's Rebuttal:
"2-hour expiry recommended"

Grounds:
- Reference to Google OAuth 2.0 Best Practices
- Avoiding increased server load from frequent reauthentication
- Minimizing user experience disruption

Concerns:
- 15-minute reauthentication increases API load by 8x
- Frequent disconnections in mobile environments

Success Metrics:
- Maintain API response time < 200ms
- Server CPU usage < 60%

Mutual Discussion:
Security → Performance:
"Business losses from security breaches are greater than server load.
Example: Equifax breach cost $700 million"

Performance → Security:
"Both can be achieved with refresh token mechanism.
Background updates ensure security without compromising UX"

Security → Performance:
"Refresh tokens are also attack targets. Proper implementation is prerequisite"

Performance → Security:
"Propose phased approach. 30 minutes for normal operations, 15 minutes for sensitive operations"

Compromise Search:
Common Understanding:
- Need to balance user experience and security
- Flexible response based on risk level
- Practical consideration of implementation and operational costs

Win-Win Elements:
- Utilization of refresh token mechanism
- Phased introduction of risk-based authentication
- Complementation with auto-logout function

Integrated Conclusion:
"30-minute expiry + refresh token + risk-based authentication"

Implementation Details:
1. Access token: 30-minute expiry
2. Refresh token: 7-day expiry
3. High-risk operations: Enforce reauthentication every 15 minutes
4. Auto-logout after 30 minutes of inactivity

Phased Implementation:
Weeks 1-2: Basic 30-minute token implementation
Weeks 3-4: Add refresh token mechanism
Month 2: Introduce risk-based authentication

Success Metrics:
- Security: Incident rate < 0.1%
- Performance: API load increase < 20%
- UX: User satisfaction > 85%

Future Review:
- After 3 months: Evaluate actual attack patterns and load conditions
- After 6 months: Consider migration to more sophisticated risk-based authentication
```

### For 3-Role Debate

```
Role Debate: Architect vs Security vs Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Topic: Pros and Cons of Microservices

Architect Role's Argument:
"Phased microservices recommended"
Grounds: Clear domain boundaries, independent deployment, freedom in technology selection

Security Role's Concerns:
"Security complexity of inter-service communication"
Grounds: Management costs of API Gateway, mTLS, distributed authentication

Performance Role's Concerns:
"Latency increase due to network communication"
Grounds: N+1 problem from internal API calls, distributed transactions

Three-Party Discussion:
Architect → Security: "Can be controlled through centralized API Gateway management"
Security → Architect: "Risk of single point of failure"
Performance → Architect: "Service division granularity is important"
...(discussion continues)

Integrated Conclusion:
"Domain-driven design for phased division + security-first design"
```

### Effective Debate Patterns

### Technology Selection

```bash
/role-debate architect,performance
"Database Selection: PostgreSQL vs MongoDB"

/role-debate frontend,mobile
"UI Framework: React vs Vue"

/role-debate security,architect
"Authentication Method: JWT vs Session Cookie"
```

### Design Decisions

```bash
/role-debate security,frontend
"User Authentication UX Design"

/role-debate performance,mobile
"Data Synchronization Strategy Optimization"

/role-debate architect,qa
"Test Strategy and Architecture Design"
```

### Trade-off Issues

```bash
/role-debate security,performance
"Encryption Level vs Processing Speed"

/role-debate frontend,performance
"Rich UI vs Page Loading Speed"

/role-debate mobile,security
"Convenience vs Data Protection Level"
```

### Role-Specific Debate Characteristics

#### 🔒 Security Role

```yaml
debate_stance:
  - Conservative approach (risk minimization)
  - Compliance-focused (cautious about deviations from standards)
  - Worst-case scenario assumption (attacker perspective)
  - Long-term impact focus (security as technical debt)

typical_issues:
  - "Security vs Convenience" trade-offs
  - "Mandatory compliance requirements"
  - "Attack cost vs Defense cost comparison"
  - "Thorough privacy protection"

evidence_sources:
  - OWASP guidelines
  - NIST frameworks
  - Industry standards (ISO 27001, SOC 2)
  - Actual attack cases and statistics

debate_strengths:
  - Precision in risk assessment
  - Knowledge of regulatory requirements
  - Understanding of attack methods

potential_biases:
  - Excessive conservatism (inhibiting innovation)
  - Insufficient UX consideration
  - Downplaying implementation costs
```

#### ⚡ Performance Role

```yaml
debate_stance:
  - Data-driven decisions (measurement-based)
  - Efficiency-focused (optimizing cost-effectiveness)
  - User experience priority (perceived speed focus)
  - Continuous improvement (phased optimization)

typical_issues:
  - "Performance vs Security"
  - "Optimization cost vs effectiveness ROI"
  - "Current vs future scalability"
  - "User experience vs system efficiency"

evidence_sources:
  - Core Web Vitals metrics
  - Benchmark results and statistics
  - Impact data on user behavior
  - Industry performance standards

debate_strengths:
  - Quantitative evaluation ability
  - Bottleneck identification
  - Knowledge of optimization techniques

potential_biases:
  - Downplaying security
  - Insufficient maintainability consideration
  - Premature optimization
```

#### 🏗️ Architect Role

```yaml
debate_stance:
  - Long-term perspective (consideration for system evolution)
  - Balance pursuit (overall optimization)
  - Phased changes (risk management)
  - Standard compliance (preference for proven patterns)

typical_issues:
  - "Short-term efficiency vs long-term maintainability"
  - "Technical debt vs development speed"
  - "Microservices vs monolith"
  - "New technology adoption vs stability"

evidence_sources:
  - Architecture pattern collections
  - Design principles (SOLID, DDD)
  - Large-scale system cases
  - Technology evolution trends

debate_strengths:
  - Overall perspective ability
  - Knowledge of design patterns
  - Prediction of long-term impacts

potential_biases:
  - Excessive generalization
  - Conservatism toward new technologies
  - Insufficient understanding of implementation details
```

#### 🎨 Frontend Role

```yaml
debate_stance:
  - User-centered design (UX first priority)
  - Inclusive approach (diversity consideration)
  - Intuitiveness focus (minimizing learning costs)
  - Accessibility standards (WCAG compliance)

typical_issues:
  - "Usability vs Security"
  - "Design consistency vs platform optimization"
  - "Functionality vs simplicity"
  - "Performance vs rich experience"

evidence_sources:
  - UX research and usability test results
  - Accessibility guidelines
  - Design system standards
  - User behavior data

debate_strengths:
  - Representation of user perspective
  - Knowledge of design principles
  - Accessibility requirements

potential_biases:
  - Insufficient understanding of technical constraints
  - Downplaying security requirements
  - Underestimation of performance impact
```

#### 📱 Mobile Role

```yaml
debate_stance:
  - Platform specialization (considering iOS/Android differences)
  - Context adaptation (on-the-go, one-handed operation)
  - Resource constraints (battery, memory, communication)
  - Store compliance (review guidelines)

typical_issues:
  - "Native vs cross-platform"
  - "Offline support vs real-time synchronization"
  - "Battery efficiency vs functionality"
  - "Platform unification vs optimization"

evidence_sources:
  - iOS HIG / Android Material Design
  - App Store / Google Play guidelines
  - Mobile UX research
  - Device performance statistics

debate_strengths:
  - Understanding of mobile-specific constraints
  - Knowledge of platform differences
  - Touch interface design

potential_biases:
  - Insufficient understanding of web platform
  - Downplaying server-side constraints
  - Insufficient consideration for desktop environment
```

#### 🔍 Analyzer Role

```yaml
debate_stance:
  - Evidence-focused (data-first)
  - Hypothesis verification (scientific approach)
  - Structural thinking (system thinking)
  - Bias elimination (objectivity pursuit)

typical_issues:
  - "Correlation vs causation"
  - "Symptomatic treatment vs root solution"
  - "Distinction between hypothesis and fact"
  - "Short-term symptoms vs structural problems"

evidence_sources:
  - Measured data and log analysis
  - Statistical methods and analysis results
  - System thinking theory
  - Cognitive bias research

debate_strengths:
  - Logical analysis ability
  - Objectivity in evidence evaluation
  - Discovery of structural problems

potential_biases:
  - Analysis paralysis (insufficient action)
  - Perfectionism (downplaying practicality)
  - Data absolutism
```

### Debate Progression Templates

#### Phase 1: Position Statement Template

```
[Role Name]'s Recommendation:
"[Specific proposal]"

Grounds:
- [Reference to official documents/standards]
- [Empirical cases/data]
- [Professional field principles]

Expected Effects:
- [Short-term effects]
- [Medium to long-term effects]

Concerns/Risks:
- [Implementation risks]
- [Operational risks]
- [Impacts on other fields]

Success Metrics:
- [Measurable metric 1]
- [Measurable metric 2]
```

#### Phase 2: Rebuttal Template

```
Rebuttal to [Target Role]:
"[Specific rebuttal to target proposal]"

Rebuttal Grounds:
- [Overlooked perspectives]
- [Contradictory evidence/cases]
- [Concerns from professional field]

Alternative Proposal:
"[Improved proposal]"

Compromise Points:
- [Acceptable conditions]
- [Possibility of phased implementation]
```

#### Phase 3: Integrated Solution Template

```
Integrated Solution:
"[Final proposal considering all roles' concerns]"

Considerations for Each Role:
- [Security]: [How security requirements are met]
- [Performance]: [How performance requirements are met]
- [Others]: [How other requirements are met]

Implementation Roadmap:
- Phase 1 (Immediate): [Urgent response items]
- Phase 2 (Short-term): [Basic implementation]
- Phase 3 (Medium-term): [Complete implementation]

Success Metrics & Measurement Methods:
- [Integrated success metrics]
- [Measurement methods/frequency]
- [Review timing]
```

### Debate Quality Checklist

#### Evidence Quality

- [ ] References to official documents/standards
- [ ] Specific cases/data presented
- [ ] Distinction between speculation and fact
- [ ] Sources explicitly stated

#### Debate Constructiveness

- [ ] Accurate understanding of opponent's proposals
- [ ] Logical rather than emotional rebuttal
- [ ] Alternatives also presented
- [ ] Exploration of win-win possibilities

#### Implementation Feasibility

- [ ] Technical feasibility considered
- [ ] Implementation costs/duration estimated
- [ ] Phased implementation possibility considered
- [ ] Risk mitigation measures presented

#### Integration

- [ ] Impacts on other fields considered
- [ ] Pursuit of overall optimization
- [ ] Long-term perspective included
- [ ] Measurable success metrics set

### Collaboration with Claude

```bash
# Debate based on design documents
cat system-design.md
/role-debate architect,security
"Discuss security issues in this design"

# Solution debate based on problems
cat performance-issues.md
/role-debate performance,architect
"Discuss fundamental solutions to performance issues"

# Technology selection debate based on requirements
/role-debate mobile,frontend
"Discuss unified UI strategy for iOS, Android, and Web"
```

### Notes

- Debates may take time (longer for complex topics)
- With 3+ roles, discussions may diverge
- Final decisions should be made by users referencing debate results
- For urgent issues, consider single role or multi-role first
