## Check Prompt

A comprehensive collection of best practices for evaluating and improving the quality of prompts for AI Agents. It systematizes knowledge gained from actual prompt improvement processes, covering all important aspects such as ambiguity elimination, information integration, enforcement enhancement, tracking systems, and continuous improvement.

### Usage

```bash
# Check the quality of a prompt file
cat your-prompt.md
/check-prompt
"Check the quality of this prompt and suggest improvements"
```

### Options

- None: Analyze current file or selected text
- `--category <name>`: Check only specific category (structure/execution/restrictions/quality/roles/improvement)
- `--score`: Calculate quality score only
- `--fix`: Automatically suggest fixes for detected issues
- `--deep`: Deep analysis mode (focus on ambiguity, information dispersion, and enforcement)

### Basic Examples

```bash
# Evaluate overall prompt quality
cat devin/playbooks/code-review.md
/check-prompt
"Evaluate this prompt across 6 categories and suggest improvements"

# Deep analysis mode
/check-prompt --deep
"Focus on checking ambiguity, information dispersion, and lack of enforcement to suggest fundamental improvements"

# Check specific category
/check-prompt --category structure
"Check this prompt from the perspective of structure and clarity"

# Detect and fix ambiguous expressions
/check-prompt --fix
"Detect ambiguous expressions and suggest corrections for clarity"
```

---

## Core Design Principles

### Principle 1: Completely Eliminate Room for Interpretation

- **Absolutely Prohibited**: "In principle", "Recommended", "If possible", "Depending on the situation", "Use your judgment"
- **Must Use**: "Always", "Absolutely", "Strictly observe", "Without exception", "Mandatory"
- **Exception Conditions**: Strictly limited by numbers ("Only under the following 3 conditions", "Except in these 2 cases")

### Principle 2: Strategic Integration of Information

- Completely integrate related important information into one section
- Summarize the overall picture in an execution checklist
- Thoroughly eliminate circular references and dispersion

### Principle 3: Building Gradual Enforcement

- Clear hierarchy of 🔴 (Execution stop level) → 🟡 (Quality important) → 🟢 (Recommended items)
- Gradual upgrade from recommended to mandatory level
- Explicit indication of impact and countermeasures for violations

### Principle 4: Ensuring Traceability

- All execution results can be recorded and verified
- Technically prevent false reporting
- Objective criteria for success/failure judgment

### Principle 5: Feedback-Driven Improvement

- Learn from actual failure cases
- Continuous effectiveness verification
- Automatic detection of new patterns

---

## 📋 Comprehensive Check Items

### 1. 📐 Structure and Clarity (Weight: 25 points)

#### 1.1 Priority Indication of Instructions (8 points)

- [ ] 🔴🟡🟢 priorities are clearly indicated for all important instructions
- [ ] Conditions for execution stop level are specifically and clearly defined
- [ ] Criteria for each priority level are objective and verifiable
- [ ] Priority hierarchy is consistently applied

#### 1.2 Complete Elimination of Ambiguous Expressions (9 points)

- [ ] **Fatal ambiguous expressions**: 0 instances of "In principle", "Recommended", "If possible"
- [ ] **Use of mandatory expressions**: Appropriate use of "Always", "Absolutely", "Strictly observe", "Without exception"
- [ ] **Numerical limitation of exception conditions**: Clear boundaries like "Only 3 conditions"
- [ ] **Elimination of judgment room**: Use only expressions that cannot be multiple interpreted
- [ ] **Elimination of gray zones**: Clear judgment criteria for all situations

#### 1.3 Strategic Integration of Information (8 points)

- [ ] Multiple location dispersion of important information is completely eliminated
- [ ] Related instructions are logically integrated into one section
- [ ] The overall picture is completely summarized in the execution checklist
- [ ] There are no circular references or infinite loops

### 2. 🎯 Executability (Weight: 20 points)

#### 2.1 Completeness of Specific Procedures (7 points)

- [ ] All command examples are actually executable and verified
- [ ] Environment variables, prerequisites, and dependencies are clearly stated without omissions
- [ ] Error handling methods are specific and executable
- [ ] The order of procedures is logical and necessary

#### 2.2 Ensuring Verifiability (7 points)

- [ ] Success/failure of execution results can be objectively determined
- [ ] Output examples, log formats, and expected values are specifically shown
- [ ] Testing methods and verification procedures can be implemented
- [ ] Checkpoints for confirming intermediate results are appropriately placed

#### 2.3 Automation Adaptability (6 points)

- [ ] Format that allows easy scripting and CI/CD integration
- [ ] Clear separation between human judgment and AI execution points
- [ ] Support for batch processing and parallel execution

### 3. 🚫 Clarification of Prohibited Items (Weight: 15 points)

#### 3.1 Systematization of Absolute Prohibitions (8 points)

- [ ] Complete list of operations that must not be performed
- [ ] Explicit indication of impact level (minor/major/fatal) for each prohibited item violation
- [ ] Specific presentation of alternatives and avoidance methods
- [ ] Explanation of technical basis for prohibited items

#### 3.2 Strict Limitation of Exception Conditions (7 points)

- [ ] Conditions allowing exceptions are specific and limited (numerical specification)
- [ ] Objective judgment criteria such as "Completely duplicate", "Explicitly stated"
- [ ] Clear boundaries without leaving gray zones
- [ ] Explicit indication of additional conditions and constraints when applying exceptions

### 4. 📊 Quality Assurance Mechanisms (Weight: 20 points)

#### 4.1 Completeness of Tracking System (8 points)

- [ ] Automatic recording and statistics collection function for all execution results
- [ ] Verification function to technically prevent false reporting
- [ ] Real-time monitoring and alert functions
- [ ] Audit log tampering prevention function

#### 4.2 Enforcement of Template Compliance (7 points)

- [ ] Clear definition and checking function for mandatory elements
- [ ] Technical restrictions on areas prohibited from customization
- [ ] Automated checkpoints for compliance confirmation
- [ ] Automatic correction and warning functions when violations occur

#### 4.3 comprehensiveness of Error Handling (5 points)

- [ ] Complete cataloging of expected error patterns
- [ ] Step-by-step handling process for errors
- [ ] Technical prevention of reporting failures as successes

### 5. 🎭 Clarification of Roles and Responsibilities (Weight: 10 points)

#### 5.1 AI Agent's Authority Scope (5 points)

- [ ] Clear boundaries between executable and prohibited operations
- [ ] Specific scope and constraints of judgment authority
- [ ] Clear separation of operations requiring human confirmation

#### 5.2 Unification of Classification System (5 points)

- [ ] Clarity, uniqueness, and exclusivity of classification definitions
- [ ] Explicit explanations to prevent misunderstanding of importance between classifications
- [ ] Specific usage examples and decision flowcharts for each classification

### 6. 🔄 Continuous Improvement (Weight: 10 points)

#### 6.1 Automation of Feedback Collection (5 points)

- [ ] Automatic extraction of improvement points from execution logs
- [ ] Machine learning-based analysis of failure patterns
- [ ] Automatic update mechanism for best practices

#### 6.2 Implementing Learning Functions (5 points)

- [ ] Automatic detection and classification of new patterns
- [ ] Continuous monitoring of effectiveness of existing rules
- [ ] Automatic suggestions for gradual improvements

---

## 🚨 Fatal Problem Patterns (Immediate Correction Required)

### ❌ Level 1: Fatal Ambiguity (Execution Stop Level)

- **Instructions with multiple interpretations**: "Use your judgment", "Depending on the situation", "In principle"
- **Ambiguous exception conditions**: "In special cases", "As needed"
- **Subjective judgment criteria**: "Appropriately", "Sufficiently", "As much as possible"
- **Undefined important concepts**: "Standard", "General", "Basic"

### ❌ Level 2: Structural Defects (Quality Important Level)

- **Information dispersion**: Important related information scattered in 3 or more locations
- **Circular references**: Infinite loops of section A→B→C→A
- **Contradictory instructions**: Contradictory instructions in different sections
- **Unclear execution order**: Procedures with unclear dependencies

### ❌ Level 3: Quality Degradation (Recommended Improvement Level)

- **Non-verifiability**: Unclear criteria for success/failure judgment
- **Difficulty in automation**: Design dependent on human subjective judgment
- **Difficulty in maintenance**: Structure where impact range during updates cannot be predicted
- **Difficulty in learning**: Complexity that takes time for newcomers to understand

---

## 🎯 Proven Improvement Methods

### ✅ Gradual Enhancement Approach

1. **Current situation analysis**: Classification, prioritization, and impact assessment of problems
2. **Fatal problem priority**: Top priority on complete resolution of Level 1 problems
3. **Gradual implementation**: Implement in verifiable units without making all changes at once
4. **Effect measurement**: Quantitative comparison before and after improvement
5. **Continuous monitoring**: Confirmation of sustainability of improvement effects

### ✅ Practical Methods for Ambiguity Elimination

```markdown
# ❌ Before Improvement (Ambiguous)

"Comments should be, in principle, written as inline comments at the corresponding change points on GitHub"

# ✅ After Improvement (Clear)

"Comments must be written as inline comments at the corresponding change points on GitHub. Exceptions are only the 3 conditions defined in section 3.3"
```

### ✅ Practical Methods for Information Integration

```markdown
# ❌ Before Improvement (Dispersed)

Section 2.1: "Use mandatory 6 sections"
Section 3.5: "📊 Comprehensive evaluation, 📋 Comments..."
Section 4.2: "Prohibition of section deletion"

# ✅ After Improvement (Integrated)

Execution Checklist:
□ 10. Post summary comment (using mandatory 6 sections)
🔴 Mandatory 6 sections: 1) 📊 Comprehensive evaluation 2) 📋 Classification of comments 3) ⚠️ Main concerns 4) ✅ Evaluable points 5) 🎯 Conclusion 6) 🤖 Self-evaluation of AI review quality
❌ Absolutely prohibited: Section deletion, addition, name change
```

### ✅ Implementation Patterns for Tracking Systems

```bash
# Strict tracking of execution results
POSTED_COMMENTS=0
FAILED_COMMENTS=0
TOTAL_COMMENTS=0

# Record results of each operation
if [ $? -eq 0 ]; then
    echo "✅ Success: $OPERATION" >> /tmp/execution_log.txt
    POSTED_COMMENTS=$((POSTED_COMMENTS + 1))
else
    echo "❌ Failure: $OPERATION" >> /tmp/execution_log.txt
    FAILED_COMMENTS=$((FAILED_COMMENTS + 1))
fi

# Prevent false reporting
if [ $POSTED_COMMENTS -ne $REPORTED_COMMENTS ]; then
    echo "🚨 Warning: Mismatch between reported and actual posted comments"
    exit 1
fi
```

---

## 📈 Quality Score Calculation (Improved Version)

### Comprehensive Score Calculation

```
Basic score = Σ(category score × weight) / 100

Fatal problem penalties:
- Level 1 problem: -20 points per case
- Level 2 problem: -10 points per case
- Level 3 problem: -5 points per case

Bonus elements:
- Automation support: +5 points
- Learning function implementation: +5 points
- Proven improvement cases: +5 points

Final score = Basic score + Bonus - Penalties
```

### Quality Level Assessment

```
95-100 points: World's highest standard (Recommended as industry standard)
90-94 points: Excellent (Production ready)
80-89 points: Good (Ready for operation with minor improvements)
70-79 points: Average (Needs improvement)
60-69 points: Needs improvement (Requires significant correction)
50-59 points: Needs major correction (Requires fundamental review)
49 points or below: Prohibited from use (Requires complete redesign)
```

---

## 🔧 Practical Improvement Process

### Phase 1: Diagnosis/Analysis (1-2 days)

1. **Understanding overall structure**: Visualization of section composition, information flow, and dependencies
2. **Ambiguity detection**: Extraction of all expressions with room for interpretation
3. **Information dispersion analysis**: Mapping of scattered patterns of related information
4. **Enforcement evaluation**: Evaluation of recommended/mandatory classification and effectiveness
5. **Traceability confirmation**: Evaluation of execution result recording and verification functions

### Phase 2: Prioritization/Planning (Half a day)

1. **Fatality classification**: Problem classification into Levels 1-3 and impact assessment
2. **Improvement order determination**: Optimal order considering interdependencies
3. **Resource allocation**: Optimization of balance between improvement effects and costs
4. **Risk assessment**: Prediction of side effects and compatibility issues during improvement

### Phase 3: Gradual Implementation (2-5 days)

1. **Level 1 problem resolution**: Complete elimination of fatal ambiguities
2. **Information integration implementation**: Strategic aggregation of dispersed information
3. **Enforcement enhancement**: Gradual upgrade from recommended to mandatory
4. **Tracking system implementation**: Automatic recording and verification functions for execution results
5. **Template enhancement**: Clarification of mandatory elements and enforcement of compliance

### Phase 4: Verification/Adjustment (1-2 days)

1. **Function testing**: Operation confirmation of all changes
2. **Integration testing**: Confirmation of system-wide consistency
3. **Performance testing**: Confirmation of execution efficiency and response
4. **Usability testing**: Verification in actual usage scenarios

### Phase 5: Operation/Monitoring (Continuous)

1. **Effect measurement**: Quantitative comparison before and after improvement
2. **Continuous monitoring**: Early detection of quality degradation
3. **Feedback collection**: Extraction of problems in actual operation
4. **Continuous optimization**: Continuous improvement cycle

---

## 📊 Actual Improvement Cases (Detailed Version)

### Case Study: Quality Improvement of Large-Scale Prompts

#### Situation Before Improvement

```bash
Quality score: 70/100 points
- Ambiguous expressions: 15 detected
- Information dispersion: Important information scattered in 6 locations
- Lack of enforcement: 80% of expressions at recommended level
- Tracking function: No execution result recording
- Error handling: Unclear countermeasures for failures
```

#### Implemented Improvements

```bash
# 1. Ambiguity elimination (2 days)
- "In principle" → "Exceptions are only the 3 conditions in section 3.3"
- "Recommended" → "Mandatory" (for importance level 2 and above)
- "As appropriate" → Explicit indication of specific judgment criteria

# 2. Information integration (1 day)
- Dispersed mandatory 6-section information → Integrated into execution checklist
- Related prohibited items → Aggregated into one section
- Eliminated circular references → Linear information flow

# 3. Tracking system implementation (1 day)
- Automatic log recording of execution results
- Verification function to prevent false reporting
- Real-time statistics display

# 4. Error handling enhancement (Half a day)
- Complete cataloging of expected error patterns
- Documentation of step-by-step handling processes
- Implementation of automatic recovery functions
```

#### Results After Improvement

```bash
Quality score: 90/100 points (+20 points improvement)
- Ambiguous expressions: 0 (completely eliminated)
- Information integration: Important information aggregated into 3 locations
- Enforcement: 95% of expressions at mandatory level
- Tracking function: Fully automated
- Error handling: 90% of problems solved automatically

Actual improvement effects:
- Assessment errors: 85% reduction
- Execution time: 40% reduction
- Error occurrence rate: 70% reduction
- User satisfaction: 95% improvement
```

### Lessons/Best Practices

#### Success Factors

1. **Gradual approach**: Implement in verifiable units without making all changes at once
2. **Data-driven**: Improve based on measured data rather than subjective judgment
3. **Continuous monitoring**: Regularly confirm the sustainability of improvement effects
4. **Feedback-oriented**: Actively collect opinions from actual users

#### Failure Avoidance Measures

1. **Excessive perfectionism**: Start operation once reaching 90 points, aim for 100 points through continuous improvement
2. **Dangers of batch changes**: Always implement large-scale changes gradually
3. **Backward compatibility**: Minimize impact on existing workflows
4. **Insufficient documentation**: Record and share all changes in detail

---

### Collaboration with Claude

```bash
# Quality check combined with prompt file
cat your-prompt.md
/check-prompt
"Evaluate the quality of this prompt and suggest improvements"

# Comparison of multiple prompt files
cat prompt-v1.md && echo "---" && cat prompt-v2.md
/check-prompt
"Compare the two versions and analyze improved points and remaining issues"

# Analysis combined with actual error logs
cat execution-errors.log
/check-prompt --deep
"Identify potential prompt issues that may have caused this error"
```

### Notes

- **Prerequisite**: Prompt files are recommended to be written in Markdown format
- **Limitation**: For large-scale prompts (10,000 lines or more), it is recommended to analyze in parts
- **Recommendation**: Regularly check prompt quality and continuously improve

---

_This checklist is a complete version of knowledge proven in actual prompt improvement projects and continues to evolve._
